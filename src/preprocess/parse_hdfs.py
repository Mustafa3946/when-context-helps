"""
Parse raw HDFS logs into structured format: (timestamp, node, message)
"""
import re
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import os
import json
from dotenv import load_dotenv

load_dotenv()

def parse_hdfs_line(line):
    """
    HDFS log format: timestamp loglevel component: message
    Example: 081109 203615 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906
    """
    pattern = r'(\d{6}\s\d{6})\s+(\d+)\s+(\w+)\s+([\w.$]+):\s+(.+)'
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    timestamp_str, node_id, level, component, message = match.groups()
    
    # Parse timestamp (YYMMDD HHMMSS)
    try:
        timestamp = datetime.strptime(timestamp_str, '%y%m%d %H%M%S').isoformat()
    except:
        timestamp = timestamp_str
    
    return {
        "timestamp": timestamp,
        "node": node_id,
        "level": level,
        "component": component,
        "message": message
    }

def preprocess_logs(connection_string=None):
    """Download from blob, parse, upload structured data."""
    if not connection_string:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Pass connection string as argument")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    
    # Download raw log
    print("Downloading raw logs from blob...")
    blob_client = blob_service.get_blob_client(container="logs-raw", blob="hdfs/HDFS_1.log")
    raw_data = blob_client.download_blob().readall().decode('utf-8')
    
    # Parse
    print("Parsing logs...")
    parsed = []
    for i, line in enumerate(raw_data.split('\n')):
        if not line.strip():
            continue
        result = parse_hdfs_line(line)
        if result:
            result["log_id"] = f"hdfs_{i}"
            parsed.append(result)
    
    print(f"Parsed {len(parsed)} log entries")
    
    # Upload structured
    output_blob = blob_service.get_blob_client(container="logs-raw", blob="hdfs/parsed.jsonl")
    output_data = '\n'.join(json.dumps(entry) for entry in parsed)
    output_blob.upload_blob(output_data, overwrite=True)
    print("Uploaded parsed logs to blob")

if __name__ == "__main__":
    import sys
    conn_str = sys.argv[1] if len(sys.argv) > 1 else None
    preprocess_logs(conn_str)
