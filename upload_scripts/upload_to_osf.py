#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA to Open Science Framework
"""
import os
import requests
import json

OSF_TOKEN = os.environ.get("OSF_TOKEN", "")
if not OSF_TOKEN:
    raise SystemExit("OSF_TOKEN environment variable is required")
BASE_URL = "https://api.osf.io/v2"

headers = {
    "Authorization": f"Bearer {OSF_TOKEN}",
    "Content-Type": "application/json"
}

# Create project
metadata = {
  "data": {
    "type": "nodes",
    "attributes": {
      "title": "ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic Retrieval-Augmented Generation",
      "category": "project",
      "description": "This repository contains the complete implementation of ASIC-RAG-CHIMERA,\n        a novel architecture that repurposes obsolete Bitcoin mining ASIC hardware\n        for cryptographically-secured Retrieval-Augmented Generation (RAG) systems.\n\n        Key features:\n        - SHA-256 hardware acceleration for cryptographic indexing\n        - AES-256-GCM encryption for data-at-rest protection\n        - Merkle tree verification for blockchain-like integrity\n        - 51,319 queries per second throughput\n        - 53/53 tests passing",
      "tags": [
        "Retrieval-Augmented Generation",
        "RAG",
        "Bitcoin ASIC",
        "Hardware Security",
        "SHA-256",
        "Cryptography",
        "Blockchain",
        "Merkle Tree",
        "Neuromorphic Computing",
        "OCaml",
        "Python",
        "Privacy-Preserving AI",
        "Deep Learning",
        "NLP",
        "Hardware Acceleration"
      ],
      "public": true
    }
  }
}

r = requests.post(
    f"{BASE_URL}/nodes/",
    headers=headers,
    data=json.dumps(metadata)
)

project_id = r.json()["data"]["id"]
print(f"[OK] Created OSF project: {project_id}")

# Get storage provider
r = requests.get(
    f"{BASE_URL}/nodes/{project_id}/files/",
    headers=headers
)
storage_provider = r.json()["data"][0]["id"]

# Upload files
files_to_upload = [
    "publication_packages/ASIC-RAG-CHIMERA_Complete_*.zip",
    "ASIC-RAG-CHIMERA_Unified.pdf",
    "README.md"
]

import glob
for filepath in files_to_upload:
    for file in glob.glob(filepath):
        filename = file.split("/")[-1]

        upload_url = f"{BASE_URL}/nodes/{project_id}/files/{storage_provider}/"

        with open(file, "rb") as fp:
            r = requests.put(
                upload_url + filename,
                headers={"Authorization": f"Bearer {OSF_TOKEN}"},
                data=fp
            )

        print(f"[OK] Uploaded: {filename}")

print(f"[OK] Project URL: https://osf.io/{project_id}/")
print("Make the project PUBLIC on OSF website!")
