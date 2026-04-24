#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA to Zenodo
"""
import os
import requests
import json

ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN", "")
if not ZENODO_TOKEN:
    raise SystemExit("ZENODO_TOKEN environment variable is required")
ZENODO_URL = "https://zenodo.org/api/deposit/depositions"

headers = {"Content-Type": "application/json"}
params = {"access_token": ZENODO_TOKEN}

# Create new deposition
r = requests.post(
    ZENODO_URL,
    params=params,
    json={},
    headers=headers
)
deposition_id = r.json()["id"]
bucket_url = r.json()["links"]["bucket"]

print(f"Created deposition: {deposition_id}")

# Upload files
files_to_upload = [
    "publication_packages/ASIC-RAG-CHIMERA_Complete_*.zip",
    "ASIC-RAG-CHIMERA_Unified.pdf",
    "README.md"
]

for filepath in files_to_upload:
    import glob
    for file in glob.glob(filepath):
        filename = file.split("/")[-1]
        with open(file, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{filename}",
                data=fp,
                params=params
            )
        print(f"[OK] Uploaded: {filename}")

# Add metadata
metadata = {
  "metadata": {
    "title": "ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic Retrieval-Augmented Generation",
    "upload_type": "dataset",
    "description": "This repository contains the complete implementation of ASIC-RAG-CHIMERA,\n        a novel architecture that repurposes obsolete Bitcoin mining ASIC hardware\n        for cryptographically-secured Retrieval-Augmented Generation (RAG) systems.\n\n        Key features:\n        - SHA-256 hardware acceleration for cryptographic indexing\n        - AES-256-GCM encryption for data-at-rest protection\n        - Merkle tree verification for blockchain-like integrity\n        - 51,319 queries per second throughput\n        - 53/53 tests passing",
    "creators": [
      {
        "name": "Francisco Angulo de Lafuente",
        "affiliation": "Independent Researcher, Advanced AI Systems Laboratory"
      }
    ],
    "keywords": [
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
    "license": "mit-license",
    "access_right": "open",
    "related_identifiers": [
      {
        "identifier": "https://github.com/Agnuxo1/ASIC-RAG-CHIMERA",
        "relation": "isSupplementTo",
        "scheme": "url"
      }
    ]
  }
}

r = requests.put(
    f"{ZENODO_URL}/{deposition_id}",
    params=params,
    data=json.dumps(metadata),
    headers=headers
)

print("[OK] Metadata added")
print(f"Deposition URL: https://zenodo.org/deposit/{deposition_id}")
print("Remember to PUBLISH the deposition manually on Zenodo website!")
