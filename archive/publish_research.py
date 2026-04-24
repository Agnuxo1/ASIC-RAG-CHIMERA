"""
Research Publication Automation Script

This script automates the process of publishing the ASIC-RAG-CHIMERA project to:
1. Zenodo (University/Archival Standard)
2. OSF (Open Science Framework)

Usage:
    python publish_research.py
"""

import requests
import os
import json
import shutil
import zipfile
from typing import Dict, Optional

# Configuration - API Keys loaded from environment (never hardcode)
ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN", "")
OSF_TOKEN = os.environ.get("OSF_TOKEN", "")

# Metadata
TITLE = "ASIC-RAG-CHIMERA: Consciousness Emergence as Phase Transition in GPU-Native Neuromorphic Computing"
DESCRIPTION = """
This project explores the emergence of consciousness as a phase transition in GPU-native neuromorphic computing,
integrated with an ASIC-accelerated Retrieval-Augmented Generation (RAG) system.
It combines high-performance hash engines with biological plausibility to study cognitive architectures.
"""
CREATORS = [{"name": "Angulo Lafuente, Francisco", "affiliation": "Independent Researcher"}]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_PATH = os.path.join(PROJECT_ROOT, "ASIC_RAG_CHIMERA_Paper.html")

def create_zip_archive(output_filename="asic_rag_repository.zip"):
    """Zips the current directory, excluding git and pycache."""
    print("Creating repository archive...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Exclude hidden dirs and the zip file itself
            dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
            if output_filename in files:
                files.remove(output_filename)
                
            for file in files:
                if file.startswith('.'): continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, PROJECT_ROOT)
                zipf.write(file_path, arcname)
    print(f"Archive created: {output_filename}")
    return output_filename

class ZenodoPublisher:
    def __init__(self, token):
        self.msg = "Use sandbox for testing, but user gave production key. Defaulting to production."
        self.base_url = "https://zenodo.org/api" # Switch to sandbox.zenodo.org for testing if needed
        self.headers = {"Authorization": f"Bearer {token}"}

    def create_deposit(self):
        url = f"{self.base_url}/deposit/depositions"
        data = {
            "metadata": {
                "title": TITLE,
                "upload_type": "software",
                "description": DESCRIPTION,
                "creators": CREATORS
            }
        }
        r = requests.post(url, json=data, headers=self.headers)
        if r.status_code != 201:
            print(f"Zenodo Create Failed: {r.status_code} - {r.text}")
            return None
        return r.json()

    def upload_file(self, deposition_id, bucket_url, file_path):
        filename = os.path.basename(file_path)
        url = f"{bucket_url}/{filename}"
        print(f"Uploading {filename} to Zenodo...")
        with open(file_path, "rb") as fp:
            r = requests.put(url, data=fp, headers=self.headers)
            if r.status_code != 200:
                print(f"Zenodo Upload Failed: {r.status_code} - {r.text}")
                return False
        return True

    def publish(self, deposition_id):
        # We generally DO NOT automate the final 'publish' action to avoid accidental public release
        # without final review. We leave it as a draft.
        print(f"Deposit created (ID: {deposition_id}). Please review and publish manually.")
        print(f"Link: https://zenodo.org/deposit/{deposition_id}")

class OSFPublisher:
    def __init__(self, token):
        self.base_url = "https://api.osf.io/v2"
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def create_node(self):
        url = f"{self.base_url}/nodes/"
        data = {
            "data": {
                "type": "nodes",
                "attributes": {
                    "title": TITLE,
                    "category": "project",
                    "description": DESCRIPTION,
                    "public": False # Create as private first
                }
            }
        }
        r = requests.post(url, json=data, headers=self.headers)
        if r.status_code != 201:
            print(f"OSF Create Node Failed: {r.status_code} - {r.text}")
            return None
        return r.json()['data']

    def upload_file(self, node_id, file_path):
        # 1. Get storage providers (osfstorage)
        url = f"{self.base_url}/nodes/{node_id}/files/osfstorage"
        # OSF upload is tricky, we upload to the root folder of osfstorage
        
        filename = os.path.basename(file_path)
        upload_url = f"https://files.osf.io/v1/resources/{node_id}/providers/osfstorage/?kind=file&name={filename}"
        
        print(f"Uploading {filename} to OSF...")
        with open(file_path, 'rb') as fp:
            r = requests.put(upload_url, data=fp, headers=self.headers)
            if r.status_code not in [200, 201]:
                 print(f"OSF Upload Failed: {r.status_code} - {r.text}")
                 return False
        return True

def main():
    print("Starting Research Publication Process...")
    
    # 1. Prepare Artifacts
    zip_path = create_zip_archive()
    files_to_upload = [zip_path]
    if os.path.exists(PAPER_PATH):
        files_to_upload.append(PAPER_PATH)
    else:
        print(f"Warning: Paper not found at {PAPER_PATH}")

    # 2. Zenodo
    print("\n--- Zenodo Publishing ---")
    zenodo = ZenodoPublisher(ZENODO_TOKEN)
    deposit = zenodo.create_deposit()
    if deposit:
        bucket_url = deposit["links"]["bucket"]
        dep_id = deposit["id"]
        for f in files_to_upload:
            zenodo.upload_file(dep_id, bucket_url, f)
        zenodo.publish(dep_id)
        
    # 3. OSF
    print("\n--- OSF Publishing ---")
    osf = OSFPublisher(OSF_TOKEN)
    node = osf.create_node()
    if node:
        node_id = node['id']
        print(f"OSF Project Created: https://osf.io/{node_id}/")
        for f in files_to_upload:
            osf.upload_file(node_id, f)
            
    # Cleanup
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("\nCleanup complete.")

if __name__ == "__main__":
    main()
