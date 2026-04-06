#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA to DataHub
"""
import json
import os
import subprocess
from pathlib import Path

# DataHub uses the 'data' CLI tool
# Install with: pip install data

def create_datapackage():
    """Create DataHub-compatible datapackage.json"""
    datapackage = {
        "name": "asic-rag-chimera",
        "title": "ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic RAG",
        "description": """
        Complete implementation and benchmarks of ASIC-RAG-CHIMERA, a novel system that
        repurposes obsolete Bitcoin mining ASICs for cryptographically-secured
        Retrieval-Augmented Generation.

        Features:
        - SHA-256 hardware acceleration (51,319 QPS)
        - AES-256-GCM encryption
        - Merkle tree integrity verification
        - 53/53 tests passing

        Dataset includes:
        - Complete source code
        - Benchmark results and performance metrics
        - Academic paper (PDF)
        - Documentation and examples
        """,
        "version": "1.0.0",
        "license": "MIT",
        "homepage": "https://github.com/Agnuxo1/ASIC-RAG-CHIMERA",
        "keywords": [
            "rag", "retrieval-augmented-generation", "cryptography",
            "hardware-acceleration", "bitcoin", "asic", "sha256",
            "encryption", "blockchain", "merkle-tree", "security"
        ],
        "contributors": [
            {
                "title": "Francisco Angulo de Lafuente",
                "email": "contact@example.com",
                "role": "author"
            }
        ],
        "sources": [
            {
                "name": "GitHub Repository",
                "web": "https://github.com/Agnuxo1/ASIC-RAG-CHIMERA"
            }
        ],
        "resources": [
            {
                "name": "benchmark_summary",
                "path": "benchmark_summary.json",
                "format": "json",
                "mediatype": "application/json",
                "description": "Comprehensive benchmark results"
            },
            {
                "name": "complete_package",
                "path": "ASIC-RAG-CHIMERA_Complete.zip",
                "format": "zip",
                "mediatype": "application/zip",
                "description": "Complete source code and documentation"
            },
            {
                "name": "paper",
                "path": "ASIC-RAG-CHIMERA_Unified.pdf",
                "format": "pdf",
                "mediatype": "application/pdf",
                "description": "Academic paper"
            }
        ]
    }

    with open('datapackage.json', 'w') as f:
        json.dump(datapackage, f, indent=2)

    print("[OK] datapackage.json created")

def prepare_datahub_directory():
    """Prepare directory structure for DataHub"""
    datahub_dir = Path('datahub_upload')
    datahub_dir.mkdir(exist_ok=True)

    # Copy files
    import shutil
    files_to_copy = [
        'publication_results/benchmark_summary.json',
        'ASIC-RAG-CHIMERA_Unified.pdf',
        'README.md'
    ]

    for file_path in files_to_copy:
        if Path(file_path).exists():
            shutil.copy(file_path, datahub_dir)
            print(f"[OK] Copied: {file_path}")

    # Copy latest complete package
    import glob
    complete_packages = glob.glob('publication_packages/ASIC-RAG-CHIMERA_Complete_*.zip')
    if complete_packages:
        latest = max(complete_packages, key=os.path.getctime)
        shutil.copy(latest, datahub_dir / 'ASIC-RAG-CHIMERA_Complete.zip')
        print(f"[OK] Copied: {latest}")

    # Move datapackage.json to directory
    shutil.move('datapackage.json', datahub_dir / 'datapackage.json')

    return datahub_dir

def validate_and_publish():
    """Validate and publish to DataHub"""
    datahub_dir = prepare_datahub_directory()
    os.chdir(datahub_dir)

    print("\n[INFO] Validating datapackage...")
    result = subprocess.run(['data', 'validate', '.'], capture_output=True, text=True)
    print(result.stdout)

    if result.returncode == 0:
        print("\n[INFO] Publishing to DataHub...")
        print("[INFO] You need to:")
        print("1. Create account at https://datahub.io/")
        print("2. Run: data login")
        print("3. Run: data push")
        print("\nManual steps required. Package ready in:", datahub_dir.absolute())
    else:
        print("[ERROR] Validation failed:", result.stderr)

if __name__ == "__main__":
    print("=" * 80)
    print("DataHub Upload Preparation")
    print("=" * 80)

    create_datapackage()
    validate_and_publish()

    print("\n[SUCCESS] DataHub package prepared!")
    print("\nNext steps:")
    print("1. Install data CLI: pip install data")
    print("2. Login: data login")
    print("3. Navigate to datahub_upload/")
    print("4. Publish: data push")
