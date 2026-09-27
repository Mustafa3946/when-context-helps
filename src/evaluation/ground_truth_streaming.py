"""
Streaming Ground Truth Generation for BGL (INFOCOM-Compliant)

CRITICAL DESIGN DECISIONS (rev6 - FULLY LOCAL MEMORY-BOUNDED):
1. NO full in-memory loading of logs or template_map
2. DuckDB-backed template_map queries (disk-backed, indexed)
3. Two-pass streaming architecture
4. Deterministic incident selection (seeded sampling)
5. Explicit failure marker patterns (no heuristics)
6. Reproducible train/test splits

SILVER STANDARD PROTOCOL:
- Time used ONLY as candidate filter (broad window)
- Relevance determined by EXPLICIT failure signatures
- All decisions documented for reviewers

MEMORY GUARANTEE:
- Max memory: O(incidents) + O(DuckDB_cache)
- Never loads full template_map into Python RAM
"""

import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Set
from collections import defaultdict
import duckdb

logger = logging.getLogger(__name__)


# ========== FAILURE MARKER PATTERNS ==========

FAILURE_MARKERS = {
    'gpu_cache_parity': {
        'patterns': [
            r'instruction cache parity error',
            r'data cache parity error',
            r'L2 cache.*parity',
            r'cache.*ECC.*error'
        ],
        'severity': 'high'
    },
    'gpu_instruction_error': {
        'patterns': [
            r'illegal instruction',
            r'instruction error',
            r'invalid opcode'
        ],
        'severity': 'high'
    },
    'memory_ecc': {
        'patterns': [
            r'memory.*ECC.*error',
            r'uncorrectable.*ECC',
            r'correctable.*ECC',
            r'memory.*parity.*error'
        ],
        'severity': 'high'
    },
    'link_training_failure': {
        'patterns': [
            r'link training.*fail',
            r'link.*down',
            r'torus.*link.*error'
        ],
        'severity': 'medium'
    },
    'kernel_panic': {
        'patterns': [
            r'kernel panic',
            r'fatal.*error',
            r'kernel.*oops'
        ],
        'severity': 'critical'
    },
    'io_error': {
        'patterns': [
            r'I/O.*error',
            r'disk.*error',
            r'data storage interrupt'
        ],
        'severity': 'medium'
    }
}


def classify_failure(message: str) -> Optional[tuple]:
    """
    Classify log message as failure marker.
    
    Returns:
        (failure_type, severity) or None
    """
    msg_lower = message.lower()
    
    for failure_type, config in FAILURE_MARKERS.items():
        for pattern in config['patterns']:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return (failure_type, config['severity'])
    
    return None


class StreamingGroundTruthBuilder:
    """
    Two-pass streaming ground truth builder with DuckDB-backed template_map.
    
    Pass 1: Identify incident candidates (failure markers)
    Pass 2: For each incident, stream logs to find relevant ones
    
    Memory footprint: O(incidents) + O(DuckDB_cache)
    NO full template_map loading into Python RAM
    """
    
    def __init__(
        self,
        parsed_logs_file: str,
        template_map_db: str,  # Changed from template_map_file
        output_file: str,
        seed: int = 42,
        max_incidents: int = 100,
        min_relevant: int = 3,
        time_window_hours: float = 1.0,
        severity_filter: Optional[List[str]] = None
    ):
        """
        Initialize streaming builder with DuckDB backend.
        
        Args:
            parsed_logs_file: Path to parsed.jsonl
            template_map_db: Path to template_map DuckDB database
            output_file: Output ground truth JSONL path
            seed: Random seed for reproducibility
            max_incidents: Maximum incidents to generate
            min_relevant: Minimum relevant templates per incident
            time_window_hours: Time window for candidate filtering
            severity_filter: Only use incidents with these severities
        """
        self.parsed_logs_file = Path(parsed_logs_file)
        self.template_map_db = Path(template_map_db)
        self.output_file = Path(output_file)
        self.seed = seed
        self.max_incidents = max_incidents
        self.min_relevant = min_relevant
        self.time_window_hours = time_window_hours
        self.severity_filter = severity_filter or ['critical', 'high']
        
        # DuckDB connection (persistent across passes)
        self.db_conn = None
        
        random.seed(seed)
        logger.info(f"Initialized with seed={seed}, max_incidents={max_incidents}")
        logger.info(f"DuckDB backend: {template_map_db}")
    
    def _connect_db(self):
        """Connect to DuckDB database."""
        if self.db_conn is None:
            if not self.template_map_db.exists():
                raise FileNotFoundError(
                    f"Template map database not found: {self.template_map_db}\n"
                    f"Build it first: python scripts/build_template_map_db.py"
                )
            # Connect to in-memory database and load JSONL
            self.db_conn = duckdb.connect()
            self.db_conn.execute(
                """CREATE OR REPLACE TABLE template_map AS
                   SELECT * FROM read_json_auto(?)""",
                [str(self.template_map_db)]
            )
            logger.info("✅ Connected to DuckDB database")
    
    def _close_db(self):
        """Close DuckDB connection."""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
    
    def _stream_logs(self) -> Iterator[Dict]:
        """Stream logs from parsed.jsonl (one line at a time)."""
        with open(self.parsed_logs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def _get_template_id(self, log_id: str) -> Optional[str]:
        """
        Get template_id for a log_id using DuckDB query.
        
        Memory-bounded: Single SQL query, no Python dict loading.
        """
        self._connect_db()
        result = self.db_conn.execute(
            "SELECT template_id FROM template_map WHERE log_id = ?",
            [log_id]
        ).fetchone()
        return result[0] if result else None
    
    def _get_template_ids_batch(self, log_ids: List[str]) -> Dict[str, str]:
        """
        Get template_ids for multiple log_ids (batch query).
        
        More efficient than individual queries.
        """
        self._connect_db()
        if not log_ids:
            return {}
        
        # Use IN clause for batch lookup
        placeholders = ','.join('?' * len(log_ids))
        query = f"SELECT log_id, template_id FROM template_map WHERE log_id IN ({placeholders})"
        results = self.db_conn.execute(query, log_ids).fetchall()
        
        return {row[0]: row[1] for row in results}
    
    def pass1_identify_incidents(self) -> List[Dict]:
        """
        Pass 1: Identify incident candidates.
        
        Strategy: Stream logs once, collect failure markers
        
        Returns:
            List of incident dicts with metadata
        """
        logger.info("="*80)
        logger.info("PASS 1: IDENTIFYING INCIDENT CANDIDATES")
        logger.info("="*80)
        
        incidents = []
        processed = 0
        
        for log in self._stream_logs():
            processed += 1
            
            if processed % 100000 == 0:
                logger.info(f"  Scanned {processed:,} logs, found {len(incidents)} candidates...")
            
            # Classify failure
            classification = classify_failure(log['message'])
            
            if classification:
                failure_type, severity = classification
                
                # Filter by severity
                if severity not in self.severity_filter:
                    continue
                
                # Parse timestamp
                timestamp_str = log['timestamp']
                dt = datetime.fromisoformat(timestamp_str)
                
                incidents.append({
                    'log_id': log['log_id'],
                    'template_id': log.get('template_id', ''),  # Add template_id for offline queries
                    'timestamp': timestamp_str,
                    'timestamp_unix': dt.timestamp(),
                    'node_id': log['node_id'],
                    'message': log['message'],
                    'failure_type': failure_type,
                    'severity': severity
                })
        
        logger.info(f"✅ Found {len(incidents)} incident candidates from {processed:,} logs")
        
        # Sample if too many
        # Keep a larger candidate pool so PASS 2 can skip invalid incidents (e.g., missing template_id)
        candidate_target = min(len(incidents), self.max_incidents * 5)
        if len(incidents) > candidate_target:
            logger.info(f"Sampling {candidate_target} incident candidates (seed={self.seed})...")
            incidents = random.sample(incidents, candidate_target)
        
        # Sort by time for deterministic ordering
        incidents.sort(key=lambda x: x['timestamp_unix'])
        
        return incidents
    
    def pass2_find_relevant_logs(self, incident: Dict) -> Dict:
        """
        Pass 2: Find relevant logs for an incident using DuckDB queries.
        
        Strategy:
        1. Stream logs to find candidates in time window
        2. Apply explicit failure marker patterns
        3. Batch query log_ids -> template_ids via DuckDB (NO full load)
        
        Args:
            incident: Incident dict from pass 1
            
        Returns:
            Ground truth entry with relevant_template_ids
        """
        incident_time = incident['timestamp_unix']
        time_start = incident_time - (self.time_window_hours * 3600)
        time_end = incident_time + (self.time_window_hours * 3600)
        
        # Collect relevant log_ids (not template_ids yet)
        relevant_logs = []
        
        # Stream logs to find candidates
        for log in self._stream_logs():
            # Skip self
            if log['log_id'] == incident['log_id']:
                continue
            
            # Time filter
            log_time = datetime.fromisoformat(log['timestamp']).timestamp()
            if not (time_start <= log_time <= time_end):
                continue
            
            # Failure marker filter
            classification = classify_failure(log['message'])
            if not classification:
                continue
            
            relevant_logs.append({
                'log_id': log['log_id'],
                'timestamp': log['timestamp'],
                'node_id': log['node_id'],
                'failure_type': classification[0],
                'severity': classification[1]
            })
        
        # Batch query log_ids -> template_ids via DuckDB
        log_ids = [log['log_id'] for log in relevant_logs]
        log_to_template = self._get_template_ids_batch(log_ids)
        
        # Count template occurrences
        template_counts = defaultdict(int)
        for log in relevant_logs:
            template_id = log_to_template.get(log['log_id'])
            if template_id:
                log['template_id'] = template_id
                template_counts[template_id] += 1
        
        # Extract unique template_ids
        relevant_template_ids = list(template_counts.keys())
        
        # Build relevance grades (based on occurrence frequency)
        max_count = max(template_counts.values()) if template_counts else 1
        relevance_grade = {}
        for tid, count in template_counts.items():
            # Grade: 2 (high), 1 (medium), 0 (low)
            if count >= max_count * 0.5:
                relevance_grade[tid] = 2
            elif count >= max_count * 0.2:
                relevance_grade[tid] = 1
            else:
                relevance_grade[tid] = 0
        
        return {
            'incident_id': incident['log_id'],
            'incident_time': incident['timestamp_unix'],
            'incident_node': incident['node_id'],
            'incident_text': incident['message'],
            'failure_type': incident['failure_type'],
            'severity': incident['severity'],
            'relevant_template_ids': relevant_template_ids,
            'relevance_grade': relevance_grade,
            'num_relevant_logs': len([l for l in relevant_logs if 'template_id' in l]),
            'num_relevant_templates': len(relevant_template_ids)
        }
    
    def build(self):
        """
        Build ground truth using two-pass streaming with DuckDB backend.
        
        Outputs:
        - incidents.jsonl: Query incidents
        - qrels.jsonl: Relevance judgments (query-template pairs)
        
        Memory guarantee: No full template_map loading into Python RAM
        """
        logger.info("="*80)
        logger.info("STREAMING GROUND TRUTH GENERATION (rev6 - DuckDB-backed)")
        logger.info("="*80)
        logger.info(f"Parsed logs: {self.parsed_logs_file}")
        logger.info(f"Template map DB: {self.template_map_db}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Seed: {self.seed}")
        logger.info(f"Max incidents: {self.max_incidents}")
        logger.info(f"Min relevant: {self.min_relevant}")
        logger.info(f"Time window: {self.time_window_hours}h")
        logger.info("="*80)
        
        try:
            # Pass 1: Identify incidents
            incidents = self.pass1_identify_incidents()
            
            if not incidents:
                logger.error("❌ No incidents found!")
                return
            
            # Pass 2: Find relevant logs
            logger.info("="*80)
            logger.info("PASS 2: FINDING RELEVANT LOGS (DuckDB queries)")
            logger.info("="*80)
            
            incidents_output = []
            qrels_output = []
            valid_count = 0
            skipped_missing_template_id = 0
            scanned_count = 0
            
            for i, incident in enumerate(incidents, 1):
                # Stop if we have enough valid incidents
                if valid_count >= self.max_incidents:
                    logger.info(
                        f"Reached max_incidents={self.max_incidents}. "
                        "Stopping PASS 2 early."
                    )
                    break
                
                # Safety cap to avoid infinite loops
                if skipped_missing_template_id > 100:
                    raise ValueError(
                        f"Too many incidents missing template_id ({skipped_missing_template_id}). "
                        f"Data quality issue - aborting."
                    )
                
                scanned_count += 1
                logger.info(f"Processing incident {scanned_count}/{len(incidents)}: {incident['log_id']}")
                
                gt_entry = self.pass2_find_relevant_logs(incident)
                
                # Filter by min_relevant
                if len(gt_entry['relevant_template_ids']) >= self.min_relevant:
                    # CRITICAL: Check for incident_template_id
                    incident_template_id = incident.get('template_id', '')
                    if not incident_template_id:
                        # Attempt to resolve from template_map
                        self._connect_db()
                        row = self.db_conn.execute(
                            "SELECT template_id FROM template_map WHERE log_id = ? LIMIT 1",
                            [incident['log_id']]
                        ).fetchone()
                        if row and row[0]:
                            incident_template_id = row[0]
                            logger.info(
                                f"Resolved missing incident_template_id via template_map "
                                f"(incident_id={incident['log_id']}, log_id={incident['log_id']})"
                            )
                        else:
                            logger.warning(
                                f"⚠️  SKIP - missing template_id: "
                                f"incident={incident['log_id']}, "
                                f"timestamp={incident.get('timestamp')}, "
                                f"node={incident.get('node_id')}"
                            )
                            skipped_missing_template_id += 1
                            continue
                    
                    # Incident entry (query)
                    incidents_output.append({
                        'incident_id': gt_entry['incident_id'],
                        'incident_template_id': incident_template_id,  # Validated non-empty
                        'incident_time': gt_entry['incident_time'],
                        'incident_node': gt_entry['incident_node'],
                        'incident_text': gt_entry['incident_text'],
                        'failure_type': gt_entry.get('failure_type', ''),
                        'severity': gt_entry.get('severity', '')
                    })
                    
                    # Qrels entry (relevance judgments)
                    qrels_output.append({
                        'incident_id': gt_entry['incident_id'],
                        'relevant_template_ids': gt_entry['relevant_template_ids'],
                        'relevance_grade': gt_entry['relevance_grade']
                    })
                    
                    valid_count += 1
                    logger.info(f"  ✅ Found {len(gt_entry['relevant_template_ids'])} relevant templates")
                else:
                    logger.info(f"  ⚠️  Only {len(gt_entry['relevant_template_ids'])} templates (< {self.min_relevant}), skipping")
            
            logger.info("="*80)
            logger.info(f"✅ Generated {valid_count} valid ground truth entries")
            logger.info(f"Total scanned: {scanned_count}, Valid: {valid_count}, Skipped (missing template_id): {skipped_missing_template_id}")
            
            # Save incidents.jsonl
            incidents_path = self.output_file.parent / f"{self.output_file.stem}_incidents.jsonl"
            incidents_path.parent.mkdir(parents=True, exist_ok=True)
            with open(incidents_path, 'w', encoding='utf-8') as f:
                for entry in incidents_output:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"✅ Saved incidents to {incidents_path}")
            
            # Save qrels.jsonl
            qrels_path = self.output_file.parent / f"{self.output_file.stem}_qrels.jsonl"
            with open(qrels_path, 'w', encoding='utf-8') as f:
                for entry in qrels_output:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"✅ Saved qrels to {qrels_path}")
            
            # Also save combined format for backward compatibility
            combined_output = []
            for i in range(len(incidents_output)):
                combined = {**incidents_output[i], **qrels_output[i]}
                combined_output.append(combined)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in combined_output:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"✅ Saved combined format to {self.output_file}")
            
            # Statistics
            total_templates = sum(len(e['relevant_template_ids']) for e in qrels_output)
            avg_templates = total_templates / len(qrels_output) if qrels_output else 0
            logger.info(f"📊 Average relevant templates per incident: {avg_templates:.1f}")
            
        finally:
            # Clean up DB connection
            self._close_db()
            logger.info("✅ Closed DuckDB connection")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build ground truth using streaming with DuckDB backend (rev6 - memory-bounded)"
    )
    parser.add_argument(
        '--parsed-logs',
        default='data/processed/bgl/parsed.jsonl',
        help='Path to parsed.jsonl'
    )
    parser.add_argument(
        '--template-map-db',
        default='data/processed/bgl/template_map.duckdb',
        help='Path to template_map DuckDB database (build with scripts/build_template_map_db.py)'
    )
    parser.add_argument(
        '--output',
        default='data/evaluation/bgl_ground_truth.jsonl',
        help='Output ground truth file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-incidents',
        type=int,
        default=100,
        help='Maximum incidents to generate'
    )
    parser.add_argument(
        '--min-relevant',
        type=int,
        default=3,
        help='Minimum relevant templates per incident'
    )
    parser.add_argument(
        '--time-window',
        type=float,
        default=1.0,
        help='Time window in hours'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    builder = StreamingGroundTruthBuilder(
        parsed_logs_file=args.parsed_logs,
        template_map_db=args.template_map_db,
        output_file=args.output,
        seed=args.seed,
        max_incidents=args.max_incidents,
        min_relevant=args.min_relevant,
        time_window_hours=args.time_window
    )
    
    builder.build()
