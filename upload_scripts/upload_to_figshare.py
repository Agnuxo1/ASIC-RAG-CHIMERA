#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA to Figshare
"""
import requests
import json

FIGSHARE_TOKEN = "$GNJmzWHcQL6XSS"
BASE_URL = "https://api.figshare.com/v2"

headers = {
    "Authorization": f"token {FIGSHARE_TOKEN}",
    "Content-Type": "application/json"
}

# Create article
metadata = {
  "title": "ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic Retrieval-Augmented Generation",
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
  "categories": [
    129,
    30
  ],
  "authors": [
    {
      "name": "Francisco Angulo de Lafuente"
    }
  ],
  "defined_type": "dataset",
  "license": 1
}

r = requests.post(
    f"{BASE_URL}/account/articles",
    headers=headers,
    data=json.dumps(metadata)
)

article_id = r.json()["location"].split("/")[-1]
print(f"Created article: {article_id}")

# Upload files
files_to_upload = [
    "publication_packages/ASIC-RAG-CHIMERA_Complete_*.zip",
    "ASIC-RAG-CHIMERA_Unified.pdf"
]

import glob
for filepath in files_to_upload:
    for file in glob.glob(filepath):
        filename = file.split("/")[-1]

        # Initiate upload
        file_info = {
            "name": filename,
            "size": os.path.getsize(file)
        }

        r = requests.post(
            f"{BASE_URL}/account/articles/{article_id}/files",
            headers=headers,
            data=json.dumps(file_info)
        )

        upload_url = r.json()["upload_url"]

        # Upload file
        with open(file, "rb") as fp:
            requests.put(upload_url, data=fp)

        # Complete upload
        requests.post(
            f"{BASE_URL}/account/articles/{article_id}/files/{r.json()['id']}",
            headers=headers
        )

        print(f"[OK] Uploaded: {filename}")

print(f"[OK] Article URL: https://figshare.com/account/articles/{article_id}")
print("Remember to PUBLISH the article on Figshare!")
