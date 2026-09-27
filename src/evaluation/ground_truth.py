"""
Ground Truth Generation for LAR-RAG Evaluation.

Implements content-based relevance protocol to avoid circular dependency with
temporal/topology priors. Uses block IDs and operation chains for relevance.
"""

import os
import json
import ssl
import re
from datetime import datetime
from collections import defaultdict
from typing import List
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.node_normalization import normalize_node_id

load_dotenv()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


def get_all_logs():
    """Retrieve all indexed logs from Azure AI Search."""
    transport = RequestsTransport(connection_verify=False)
    client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="log-embeddings",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
        transport=transport
    )
    results = list(client.search(search_text="*", top=2000, select=["log_id", "timestamp", "node", "message"]))
    print(f"Retrieved {len(results)} indexed logs")
    return results


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string."""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return datetime.strptime(ts_str, "%y%m%d %H%M%S")


def extract_block_id(message: str) -> str:
    """Extract block ID from log message."""
    match = re.search(r'blk_[-\d]+', message)
    return match.group() if match else None


def is_anomaly_marker(log: dict) -> tuple:
    """Check if log is anomaly marker. Returns (is_marker, incident_type)."""
    msg_lower = log['message'].lower()
    if 'exception' in msg_lower or 'error' in msg_lower:
        return (True, 'exception')
    if 'delet' in msg_lower and 'blk' in msg_lower:
        return (True, 'block_deletion')
    if ('receiv' in msg_lower or 'transfer' in msg_lower) and ('fail' in msg_lower or 'error' in msg_lower):
        return (True, 'receive_failure')
    if 'terminating' in msg_lower or 'shutdown' in msg_lower:
        return (True, 'node_termination')
    return (False, None)


def find_relevant_logs_content_based(incident_log: dict, all_logs: List[dict]) -> List[dict]:
    """Find relevant logs using CONTENT-BASED criteria (not time/topology)."""
    relevant = []
    incident_block = extract_block_id(incident_log['message'])
    incident_time = parse_timestamp(incident_log['timestamp'])
    incident_node = incident_log.get('node', '')
    
    # Filter candidates within 1-hour window (filtering only, not relevance criteria)
    candidates = [log for log in all_logs 
                  if log['log_id'] != incident_log['log_id'] 
                  and abs((parse_timestamp(log['timestamp']) - incident_time).total_seconds()) <= 3600]
    
    # Criterion 1: Same block ID (causal relationship - strongest signal)
    if incident_block:
        block_matches = [log for log in candidates if extract_block_id(log['message']) == incident_block]
        relevant.extend(block_matches)
    
    # Criterion 2: Same node + related operations (content-based)
    msg_lower = incident_log['message'].lower()
    operation_keywords = []
    if 'receiv' in msg_lower:
        operation_keywords = ['allocat', 'writ', 'serv', 'close', 'receiv']
    elif 'delet' in msg_lower:
        operation_keywords = ['remove', 'invalid', 'corrupt', 'delet']
    elif 'exception' in msg_lower or 'error' in msg_lower or 'fail' in msg_lower:
        operation_keywords = ['fail', 'abort', 'retry', 'exception', 'error']
    
    # Look for same component (DataNode, NameNode, etc.)
    components = ['datanode', 'namenode', 'packetresponder', 'dataxceiver']
    incident_component = None
    for comp in components:
        if comp in msg_lower:
            incident_component = comp
            break
    
    # Add logs with related operations on same component
    for log in candidates:
        if log not in relevant:
            log_lower = log['message'].lower()
            # Same component + related operations
            if incident_component and incident_component in log_lower:
                for keyword in operation_keywords:
                    if keyword in log_lower:
                        relevant.append(log)
                        break
    
    # Criterion 3: Same error type (exception matching)
    if 'exception' in msg_lower or 'error' in msg_lower:
        exception_match = re.search(r'(\w+Exception|\w+Error)', incident_log['message'])
        if exception_match:
            exception_type = exception_match.group(1)
            relevant.extend([log for log in candidates if log not in relevant and exception_type in log['message']])
    
    # Criterion 4: Same node + anomaly-related keywords (broadens coverage)
    # Only use if we don't have enough relevant logs yet
    if len(relevant) < 5 and incident_node:
        anomaly_keywords = ['exception', 'error', 'fail', 'invalid', 'corrupt', 'terminat', 'shutdown']
        for log in candidates:
            if log not in relevant and log.get('node') == incident_node:
                log_lower = log['message'].lower()
                if any(keyword in log_lower for keyword in anomaly_keywords):
                    relevant.append(log)
    
    # Remove duplicates
    seen = set()
    relevant_unique = []
    for log in relevant:
        if log['log_id'] not in seen:
            seen.add(log['log_id'])
            relevant_unique.append(log)
    
    return relevant_unique


def generate_hdfs_ground_truth(min_relevant: int = 3, target_incidents: int = 30):
    """Generate HDFS ground truth with content-based protocol."""
    print("="*80)
    print("Generating HDFS Ground Truth (Content-Based Protocol)")
    print("="*80)
    
    logs = get_all_logs()
    incidents = []
    
    # Find anomaly markers
    print("\nStep 1: Identifying anomaly markers...")
    anomaly_markers = [(log, itype) for log in logs for is_m, itype in [is_anomaly_marker(log)] if is_m]
    print(f"Found {len(anomaly_markers)} potential anomaly markers")
    
    # For each marker, find relevant logs
    print("\nStep 2: Finding relevant logs (content-based)...")
    checked = 0
    skipped_counts = []
    for marker_log, incident_type in anomaly_markers[:target_incidents * 3]:  # Check more to find enough
        checked += 1
        incident_node = normalize_node_id(marker_log['node'])
        incident_time = parse_timestamp(marker_log['timestamp'])
        
        relevant_logs = find_relevant_logs_content_based(marker_log, logs)
        
        if len(relevant_logs) < min_relevant:
            skipped_counts.append(len(relevant_logs))
            continue
        
        incident_id = f"hdfs_{incident_type}_{len(incidents) + 1:03d}"
        block_id = extract_block_id(marker_log['message'])
        query_text = f"Investigating {incident_type.replace('_', ' ')} for {block_id or 'incident'} on node {incident_node}"
        
        incidents.append({
            "incident_id": incident_id,
            "incident_time": incident_time.isoformat(),
            "incident_node": incident_node,
            "incident_text": query_text,
            "relevant_log_ids": [log['log_id'] for log in relevant_logs],
            "incident_type": incident_type,
            "dataset": "hdfs",
            "trigger_log_id": marker_log['log_id'],
            "num_relevant": len(relevant_logs),
            "relevance_rationale": f"Content-based: block_id={block_id}, operation_chain, error_correlation"
        })
        
        print(f"  ✓ Incident {len(incidents)}: {incident_type} with {len(relevant_logs)} relevant logs")
        
        if len(incidents) >= target_incidents:
            break
    
    print(f"\nChecked {checked} markers, skipped {len(skipped_counts)} (avg {sum(skipped_counts)/len(skipped_counts) if skipped_counts else 0:.1f} relevant logs)")
    
    # Save ground truth
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hdfs_ground_truth.jsonl"
    
    with open(output_file, 'w') as f:
        for incident in incidents:
            f.write(json.dumps(incident) + '\n')
    
    print(f"\n{'='*80}")
    print(f"Generated {len(incidents)} HDFS incidents")
    print(f"Saved to {output_file}")
    
    # Statistics
    type_counts = defaultdict(int)
    relevant_counts = [inc['num_relevant'] for inc in incidents]
    for inc in incidents:
        type_counts[inc['incident_type']] += 1
    
    print(f"\nIncident Types:")
    for itype, count in sorted(type_counts.items()):
        print(f"  {itype}: {count}")
    print(f"\nRelevant Logs: min={min(relevant_counts)}, max={max(relevant_counts)}, mean={sum(relevant_counts)/len(relevant_counts):.1f}")
    print("="*80)
    
    return incidents


if __name__ == "__main__":
    generate_hdfs_ground_truth(min_relevant=3, target_incidents=30)
