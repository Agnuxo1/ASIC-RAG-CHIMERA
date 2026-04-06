#!/usr/bin/env python3
"""
Upload ASIC-RAG-CHIMERA benchmarks to OpenML
"""
import openml
import json
import pandas as pd
from pathlib import Path

# Configure OpenML API key
openml.config.apikey = 'YOUR_OPENML_API_KEY'  # Get from https://www.openml.org/auth/profile-page

# Load benchmark results
with open('publication_results/benchmark_summary.json', 'r') as f:
    benchmarks = json.load(f)

# Create benchmark dataset as pandas DataFrame
benchmark_data = []

# Search latency benchmarks
for operation, metrics in benchmarks['search_latency'].items():
    benchmark_data.append({
        'category': 'search_latency',
        'operation': operation,
        'mean_ms': metrics['mean_ms'],
        'p95_ms': metrics['p95_ms'],
        'qps': metrics['qps']
    })

# Hash performance benchmarks
benchmark_data.append({
    'category': 'hash_performance',
    'operation': 'sha256_throughput',
    'value': benchmarks['hash_performance']['sha256_throughput'],
    'speedup': benchmarks['hash_performance']['speedup_vs_hashlib']
})

df = pd.DataFrame(benchmark_data)

# Save to CSV
csv_path = Path('publication_packages/asic_rag_benchmarks.csv')
df.to_csv(csv_path, index=False)

print(f"[OK] Benchmark CSV created: {csv_path}")

# Create OpenML dataset
dataset = openml.datasets.functions.create_dataset(
    name="ASIC-RAG-CHIMERA Benchmarks",
    description="""
    Comprehensive benchmark results for ASIC-RAG-CHIMERA, a hardware-accelerated
    cryptographic RAG system that repurposes obsolete Bitcoin mining ASICs.

    Includes:
    - Search latency metrics (tag lookup, AND/OR search, Merkle verification)
    - Hash performance metrics (SHA-256 throughput)
    - Full query pipeline performance

    System specs: Intel i7-12700K, 32GB RAM, RTX 3080, NVMe SSD, 2x Antminer S9
    """,
    creator="Francisco Angulo de Lafuente",
    contributor=None,
    collection_date="2024-12-09",
    language="English",
    licence="MIT",
    default_target_attribute=None,
    row_id_attribute=None,
    ignore_attribute=None,
    citation="Angulo de Lafuente, F. (2024). ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic RAG",
    attributes="auto",
    data=df
)

# Publish dataset
dataset.publish()
print(f"[OK] Dataset published to OpenML: https://www.openml.org/d/{dataset.dataset_id}")

# Add tags
openml.datasets.tag_dataset(dataset.dataset_id,
    ['rag', 'cryptography', 'hardware-acceleration', 'benchmark', 'asic', 'bitcoin'])

print("[SUCCESS] OpenML upload complete!")
