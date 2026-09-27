"""
Download HDFS log dataset and upload to Azure Blob Storage.
"""
import os
import requests
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Using LogHub HDFS dataset
HDFS_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
LOCAL_FILE = "HDFS_2k.log"

def download_hdfs():
    """Download HDFS logs if not already present."""
    if os.path.exists(LOCAL_FILE):
        print(f"{LOCAL_FILE} already exists, skipping download")
        return
    
    print(f"Downloading HDFS logs from {HDFS_URL}")
    response = requests.get(HDFS_URL, stream=True, verify=False)
    response.raise_for_status()
    
    with open(LOCAL_FILE, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {LOCAL_FILE}")

def upload_to_blob(connection_string=None):
    """Upload logs to Azure Blob Storage."""
    if not connection_string:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("ERROR: Pass connection string as argument or set AZURE_STORAGE_CONNECTION_STRING")
        return
    
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_name = "logs-raw"
    
    blob_client = blob_service.get_blob_client(container=container_name, blob="hdfs/HDFS_1.log")
    
    print(f"Uploading to Azure Blob: {container_name}/hdfs/HDFS_1.log")
    with open(LOCAL_FILE, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print("Upload complete")

if __name__ == "__main__":
    import sys
    download_hdfs()
    # Pass connection string as command line arg if .env doesn't work
    conn_str = sys.argv[1] if len(sys.argv) > 1 else None
    upload_to_blob(conn_str)
