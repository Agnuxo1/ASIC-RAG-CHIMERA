#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA to Kaggle Datasets
"""
import json
import os
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def prepare_kaggle_dataset():
    """Prepare dataset for Kaggle"""

    # Create Kaggle dataset directory
    kaggle_dir = Path('kaggle_dataset')
    kaggle_dir.mkdir(exist_ok=True)

    # Copy files
    files_to_copy = [
        'README.md',
        'ASIC-RAG-CHIMERA_Unified.pdf',
        'publication_results/benchmark_summary.json',
        'requirements.txt'
    ]

    for file_path in files_to_copy:
        if Path(file_path).exists():
            shutil.copy(file_path, kaggle_dir)
            print(f"[OK] Copied: {file_path}")

    # Copy complete package
    import glob
    complete_packages = glob.glob('publication_packages/ASIC-RAG-CHIMERA_Complete_*.zip')
    if complete_packages:
        latest = max(complete_packages, key=os.path.getctime)
        shutil.copy(latest, kaggle_dir / 'ASIC-RAG-CHIMERA_Complete.zip')
        print(f"[OK] Copied: {latest}")

    # Create dataset-metadata.json for Kaggle
    metadata = {
        "title": "ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic RAG",
        "id": "franciscoangulo/asic-rag-chimera",
        "licenses": [
            {
                "name": "MIT"
            }
        ],
        "keywords": [
            "artificial intelligence",
            "cryptography",
            "deep learning",
            "nlp",
            "python",
            "rag",
            "retrieval augmented generation",
            "hardware acceleration",
            "bitcoin",
            "asic",
            "security"
        ],
        "subtitle": "Repurposing Bitcoin Mining ASICs for Secure RAG Systems",
        "description": """
# ASIC-RAG-CHIMERA

**Hardware-Accelerated Cryptographic Retrieval-Augmented Generation System**

## Overview

This dataset contains the complete implementation, benchmarks, and documentation for ASIC-RAG-CHIMERA,
a novel system that repurposes obsolete Bitcoin mining ASIC hardware for cryptographically-secured
Retrieval-Augmented Generation.

## Key Features

- **SHA-256 Hardware Acceleration**: 51,319 queries per second
- **AES-256-GCM Encryption**: Military-grade data protection
- **Merkle Tree Verification**: Blockchain-style integrity guarantees
- **Comprehensive Testing**: 53/53 tests passing

## Performance Metrics

| Operation | Mean (ms) | P95 (ms) | QPS |
|-----------|-----------|----------|-----|
| Tag Lookup | 0.02 | 0.04 | 51,319 |
| AND Search | 0.04 | 0.06 | 24,373 |
| Full Pipeline | 47.10 | 51.90 | 21 |

## Contents

- Complete Python source code
- Benchmark results and performance metrics
- Academic paper (PDF)
- Documentation and API reference
- Example usage and demos

## Citation

```
@article{angulo2024asicrag,
  title={ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic Framework for Secure RAG},
  author={Angulo de Lafuente, Francisco},
  year={2024}
}
```

## Links

- GitHub: https://github.com/Agnuxo1/ASIC-RAG-CHIMERA
- Paper: Included in this dataset
- Documentation: See README.md
        """,
        "isPrivate": False,
        "resources": [
            {
                "path": "README.md",
                "description": "Project documentation and setup instructions"
            },
            {
                "path": "ASIC-RAG-CHIMERA_Unified.pdf",
                "description": "Academic paper with full technical details"
            },
            {
                "path": "benchmark_summary.json",
                "description": "Comprehensive benchmark results"
            },
            {
                "path": "ASIC-RAG-CHIMERA_Complete.zip",
                "description": "Complete source code package"
            }
        ]
    }

    # Save metadata
    metadata_path = kaggle_dir / 'dataset-metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Kaggle metadata created: {metadata_path}")
    return kaggle_dir

def upload_to_kaggle(dataset_dir):
    """Upload dataset to Kaggle"""

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    print("\n[INFO] Uploading to Kaggle...")

    # Create or update dataset
    try:
        api.dataset_create_new(
            folder=str(dataset_dir),
            dir_mode='zip',
            convert_to_csv=False,
            public=True
        )
        print("[SUCCESS] Dataset created on Kaggle!")
        print("Dataset URL: https://www.kaggle.com/datasets/franciscoangulo/asic-rag-chimera")
    except Exception as e:
        print(f"[INFO] Dataset might already exist, attempting update...")
        try:
            api.dataset_create_version(
                folder=str(dataset_dir),
                version_notes="Updated with latest benchmark results and documentation",
                dir_mode='zip',
                convert_to_csv=False
            )
            print("[SUCCESS] Dataset updated on Kaggle!")
            print("Dataset URL: https://www.kaggle.com/datasets/franciscoangulo/asic-rag-chimera")
        except Exception as e2:
            print(f"[ERROR] Upload failed: {e2}")
            print("\nManual upload instructions:")
            print("1. Go to https://www.kaggle.com/datasets")
            print("2. Click 'New Dataset'")
            print(f"3. Upload files from: {dataset_dir.absolute()}")

if __name__ == "__main__":
    print("=" * 80)
    print("Kaggle Dataset Upload")
    print("=" * 80)

    # Prepare dataset
    dataset_dir = prepare_kaggle_dataset()

    # Upload
    print("\n[INFO] Starting Kaggle upload...")
    print("[INFO] Make sure you have configured Kaggle API credentials:")
    print("       ~/.kaggle/kaggle.json (Linux/Mac)")
    print("       C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")

    try:
        upload_to_kaggle(dataset_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        print(f"\n[INFO] Dataset prepared in: {dataset_dir.absolute()}")
        print("[INFO] You can upload manually at: https://www.kaggle.com/datasets")

    print("\n[SUCCESS] Kaggle preparation complete!")
