"""
BGL Ground Truth Generation with Content-Based Protocol

Similar to HDFS ground truth but adapted for BGL log characteristics.
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


def get_all_bgl_logs():
    """Retrieve all indexed BGL logs from Azure AI Search."""
    transport = RequestsTransport(connection_verify=False)
    client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="bgl-logs",  # Separate index for BGL
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
        transport=transport
    )
    results = list(client.search(search_text="*", top=2000, select=["log_id", "timestamp", "node", "message"]))
    print(f"Retrieved {len(results)} indexed BGL logs")
    return results


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string."""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return datetime.strptime(ts_str, "%Y-%m-%d-%H:%M:%S")


def is_bgl_anomaly_marker(log: dict) -> tuple:
    """
    Check if BGL log is anomaly marker.
    
    BGL-specific anomaly patterns:
    - Errors and exceptions
    - Instruction errors
    - Data storage interrupts
    - Kernel panics
    - Node failures
    
    Returns (is_marker, incident_type)
    """
    msg_lower = log['message'].lower()
    
    # BGL-specific anomaly types
    if 'fatal' in msg_lower or 'panic' in msg_lower:
        return (True, 'fatal_error')
    elif 'instruction error' in msg_lower or 'illegal instruction' in msg_lower:
        return (True, 'instruction_error')
    elif 'data storage interrupt' in msg_lower:
        return (True, 'storage_interrupt')
    elif 'exception' in msg_lower or 'error' in msg_lower:
        return (True, 'exception')
    elif 'timeout' in msg_lower:
        return (True, 'timeout')
    elif 'fail' in msg_lower and not 'failure prediction' in msg_lower:
        return (True, 'failure')
    
    return (False, None)


def find_relevant_bgl_logs(incident_log: dict, all_logs: List[dict]) -> List[dict]:
    """
    Find relevant logs using CONTENT-BASED criteria for BGL.
    
    BGL relevance criteria:
    1. Same node (strong signal for HPC logs)
    2. Same error type
    3. Similar error patterns
    4. Temporal proximity (filtering only)
    """
    relevant = []
    incident_time = parse_timestamp(incident_log['timestamp'])
    incident_node = incident_log.get('node', '')
    msg_lower = incident_log['message'].lower()
    
    # Filter candidates within 1-hour window
    candidates = [log for log in all_logs 
                  if log['log_id'] != incident_log['log_id'] 
                  and abs((parse_timestamp(log['timestamp']) - incident_time).total_seconds()) <= 3600]
    
    # Criterion 1: Same node (HPC nodes typically fail together or in cascade)
    if incident_node:
        same_node_logs = [log for log in candidates if log.get('node') == incident_node]
        relevant.extend(same_node_logs[:20])  # Limit to avoid too many
    
    # Criterion 2: Same error type
    if 'exception' in msg_lower or 'error' in msg_lower:
        # Extract error type
        error_match = re.search(r'(instruction error|storage interrupt|data error|address error)', msg_lower)
        if error_match:
            error_type = error_match.group(1)
            error_matches = [log for log in candidates 
                            if log not in relevant and error_type in log['message'].lower()]
            relevant.extend(error_matches)
    
    # Criterion 3: Related error keywords
    error_keywords = []
    if 'instruction' in msg_lower:
        error_keywords = ['address', 'data', 'register', 'memory']
    elif 'storage' in msg_lower:
        error_keywords = ['interrupt', 'data', 'read', 'write']
    elif 'fatal' in msg_lower or 'panic' in msg_lower:
        error_keywords = ['kernel', 'critical', 'fatal', 'panic']
    
    for log in candidates:
        if log not in relevant:
            log_lower = log['message'].lower()
            if any(keyword in log_lower for keyword in error_keywords):
                relevant.append(log)
    
    # Criterion 4: Same rack/midplane (node proximity)
    # BGL nodes like R45-M0-N3 share rack (R) and midplane (M)
    if incident_node and '-' in incident_node:
        incident_rack = incident_node.split('-')[0] if '-' in incident_node else incident_node
        for log in candidates:
            if log not in relevant:
                log_node = log.get('node', '')
                if log_node and log_node.startswith(incident_rack):
                    # Same rack - potential cascade failure
                    if any(kw in log['message'].lower() for kw in ['error', 'fail', 'exception']):
                        relevant.append(log)
    
    # Remove duplicates
    seen = set()
    relevant_unique = []
    for log in relevant:
        if log['log_id'] not in seen:
            seen.add(log['log_id'])
            relevant_unique.append(log)
    
    return relevant_unique


def generate_bgl_ground_truth(min_relevant: int = 3, target_incidents: int = 30):
    """Generate BGL ground truth with content-based protocol."""
    print("="*80)
    print("Generating BGL Ground Truth (Content-Based Protocol)")
    print("="*80)
    
    logs = get_all_bgl_logs()
    incidents = []
    
    # Find anomaly markers
    print("\nStep 1: Identifying BGL anomaly markers...")
    anomaly_markers = [(log, itype) for log in logs for is_m, itype in [is_bgl_anomaly_marker(log)] if is_m]
    print(f"Found {len(anomaly_markers)} potential anomaly markers")
    
    # For each marker, find relevant logs
    print("\nStep 2: Finding relevant logs (content-based)...")
    checked = 0
    skipped_counts = []
    
    for marker_log, incident_type in anomaly_markers[:target_incidents * 3]:
        checked += 1
        incident_node = normalize_node_id(marker_log['node'])
        incident_time = parse_timestamp(marker_log['timestamp'])
        
        relevant_logs = find_relevant_bgl_logs(marker_log, logs)
        
        if len(relevant_logs) < min_relevant:
            skipped_counts.append(len(relevant_logs))
            continue
        
        incident_id = f"bgl_{incident_type}_{len(incidents) + 1:03d}"
        query_text = f"Investigating {incident_type.replace('_', ' ')} on node {incident_node}"
        
        incidents.append({
            "incident_id": incident_id,
            "incident_time": incident_time.isoformat(),
            "incident_node": incident_node,
            "incident_text": query_text,
            "relevant_log_ids": [log['log_id'] for log in relevant_logs],
            "incident_type": incident_type,
            "dataset": "bgl",
            "trigger_log_id": marker_log['log_id'],
            "num_relevant": len(relevant_logs),
            "relevance_rationale": f"Content-based: same_node, error_type, related_failures"
        })
        
        print(f"  Incident {len(incidents)}: {incident_type} with {len(relevant_logs)} relevant logs")
        
        if len(incidents) >= target_incidents:
            break
    
    print(f"\nChecked {checked} markers, skipped {len(skipped_counts)} (avg {sum(skipped_counts)/len(skipped_counts) if skipped_counts else 0:.1f} relevant logs)")
    
    # Save ground truth
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bgl_ground_truth.jsonl"
    
    with open(output_file, 'w') as f:
        for incident in incidents:
            f.write(json.dumps(incident) + '\n')
    
    print(f"\n{'='*80}")
    print(f"Generated {len(incidents)} BGL incidents")
    print(f"Saved to {output_file}")
    
    # Statistics
    if incidents:
        incident_types = defaultdict(int)
        relevant_counts = []
        for inc in incidents:
            incident_types[inc['incident_type']] += 1
            relevant_counts.append(inc['num_relevant'])
        
        print(f"\nIncident Types:")
        for itype, count in sorted(incident_types.items()):
            print(f"  {itype}: {count}")
        
        print(f"\nRelevant Logs: min={min(relevant_counts)}, max={max(relevant_counts)}, mean={sum(relevant_counts)/len(relevant_counts):.1f}")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BGL ground truth")
    parser.add_argument("--min-relevant", type=int, default=3, help="Minimum relevant logs per incident")
    parser.add_argument("--target-incidents", type=int, default=30, help="Target number of incidents")
    
    args = parser.parse_args()
    
    generate_bgl_ground_truth(args.min_relevant, args.target_incidents)
