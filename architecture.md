# ASIC-RAG-CHIMERA Architecture

## Overview

ASIC-RAG-CHIMERA is an innovative encrypted retrieval-augmented generation system that combines:

1. **ASIC SHA-256 Simulator**: Hardware-grade cryptographic indexing
2. **Encrypted RAG Storage**: Blockchain-inspired secure document storage
3. **CHIMERA GPU Acceleration**: Graphics pipeline for computation
4. **Local LLM Integration**: Privacy-preserving AI responses

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│                    (CLI / API / Web Interface)                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Query Engine                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  Keyword    │  │   Context    │  │     Response Generator      │ │
│  │  Extractor  │→ │   Assembler  │→ │        (Qwen3-0.6B)         │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
┌──────────────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│   ASIC Simulator     │ │   Merkle     │ │    Key Generator         │
│  ┌────────────────┐  │ │    Tree      │ │  ┌────────────────────┐  │
│  │ SHA-256 Engine │  │ │              │ │  │ Session Keys       │  │
│  │ (256 lanes)    │  │ │  O(log n)    │ │  │ (30s TTL)          │  │
│  └────────────────┘  │ │  proofs      │ │  └────────────────────┘  │
│  ┌────────────────┐  │ └──────────────┘ │  ┌────────────────────┐  │
│  │ Index Manager  │  │                  │  │ PBKDF2 Derivation  │  │
│  └────────────────┘  │                  │  └────────────────────┘  │
└──────────────────────┘                  └──────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Encrypted Block Storage                         │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ Knowledge   │  │   AES-256-GCM   │  │      LMDB Backend       │  │
│  │   Blocks    │→ │   Encryption    │→ │    (Persistent Store)   │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CHIMERA GPU Integration                         │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ GPU Hash    │  │ Texture Memory  │  │   Render Compute        │  │
│  │   Engine    │  │    Manager      │  │     Bridge              │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ASIC SHA-256 Simulator

The ASIC simulator replicates hardware-grade SHA-256 computation:

- **256 Parallel Lanes**: Simulates ASIC parallel processing
- **Batch Processing**: Up to 64K hashes per batch
- **Caching**: LRU cache for repeated hashes
- **Pipeline Architecture**: 4-stage pipeline for continuous throughput

```python
from asic_simulator import SHA256Engine

engine = SHA256Engine(num_lanes=256)
result = engine.hash_batch(data_list)
print(f"Throughput: {result.hashes_per_second:,.0f} H/s")
```

### 2. Merkle Tree Index

Hierarchical cryptographic indexing for O(log n) verification:

- **Proof Generation**: Generate inclusion proofs for any document
- **Category Trees**: Separate subtrees per document category
- **Global Root**: Single root hash for entire document set
- **Verification**: Stateless proof verification

```python
from asic_simulator import MerkleTree

tree = MerkleTree()
for doc in documents:
    tree.add_leaf(doc_hash)
tree.build()

proof = tree.get_proof(leaf_index)
assert tree.verify_proof(proof)
```

### 3. Index Manager

Tag-based document indexing with opaque keyword hashes:

- **SHA-256 Tag Hashes**: Keywords are hashed before indexing
- **Boolean Search**: AND, OR, NOT operations
- **Category Filtering**: Filter by document category
- **Relevance Scoring**: Score based on tag match ratio

```python
from asic_simulator import IndexManager, SearchOperation

index = IndexManager()
index.add_tag(tag_hash, block_id, category="finance")

result = index.search(
    tags=[tag1, tag2],
    operation=SearchOperation.AND,
    category="finance"
)
```

### 4. Knowledge Blocks

Blockchain-inspired encrypted document storage:

```
┌────────────────────────────────────────────────────────────┐
│                     Block Header (128 bytes)               │
├──────────────┬──────────────┬──────────────┬──────────────┤
│  block_hash  │  prev_hash   │  timestamp   │ category_hash│
│   (32 bytes) │  (32 bytes)  │  (8 bytes)   │  (32 bytes)  │
├──────────────┴──────────────┴──────────────┴──────────────┤
│              keywords_hash (32 bytes)                      │
├────────────────────────────────────────────────────────────┤
│              nonce (8) | flags (4) | reserved (12)         │
└────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────┐
│                     Tag Hashes                             │
│              [32 bytes × N keywords]                       │
└────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────┐
│                  Encrypted Payload                         │
│  ┌──────────┬─────────────────────────────────────────┐   │
│  │  Nonce   │              Ciphertext                  │   │
│  │(12 bytes)│    (metadata + content + auth tag)       │   │
│  └──────────┴─────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### 5. Key Generator

Temporary session key management:

- **HMAC-based Derivation**: Keys derived from master key + block hash
- **30-second TTL**: Keys expire automatically
- **Session Grouping**: Keys grouped by session
- **Automatic Cleanup**: Background thread removes expired keys

```python
from asic_simulator import KeyGenerator

generator = KeyGenerator(master_key=master_key, default_key_ttl=30)
session = generator.create_session()

key = generator.generate_key(session.session_id, block_hash)
# Key expires after 30 seconds
```

### 6. CHIMERA GPU Integration

Graphics pipeline for computation:

- **GPU Hash Engine**: CUDA-accelerated hashing
- **Texture Memory**: GPU texture storage for indices
- **Render Compute**: Fragment shaders as compute kernels

```python
from chimera_integration import GPUHashEngine

engine = GPUHashEngine(device="cuda")
result = engine.hash_batch(data_list)
```

## Data Flow

### Document Ingestion

```
Document → Chunking → Keyword Extraction → Tag Hashing
                                              │
                                              ▼
Block Creation → Encryption (AES-256-GCM) → Storage
       │                                       │
       ▼                                       ▼
  Merkle Tree ←──────────────────────── Index Update
```

### Query Processing

```
Query → Keyword Extraction → Tag Hashing
                                │
                                ▼
               Index Search (ASIC Simulator)
                                │
                                ▼
        Block Retrieval → Key Generation → Decryption
                                                │
                                                ▼
                         Context Assembly → LLM Response
```

## Security Model

### Encryption

- **Algorithm**: AES-256-GCM
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Nonce**: 12-byte random per encryption
- **Authentication**: GCM provides authenticated encryption

### Key Management

- **Master Key**: 256-bit user-provided key
- **Block Keys**: Derived via PBKDF2 from master + block hash
- **Session Keys**: Temporary keys with 30-second TTL
- **Key Rotation**: Supported via re-encryption

### Privacy

- **Opaque Tags**: Keywords are SHA-256 hashed before indexing
- **Encrypted Content**: All document content is encrypted
- **Local LLM**: No data leaves the system

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Hash Throughput | 1.2M hashes/sec (CPU) |
| Search Latency | <3ms for 10K documents |
| Encryption Speed | 500+ MB/s |
| Key Generation | 50µs per key |
| Merkle Proof | O(log n) |

## Configuration

See `config/default_config.yaml` for full configuration options.

Key settings:
- `asic_simulator.num_lanes`: Parallel hash lanes (default: 256)
- `encryption.pbkdf2_iterations`: Key derivation iterations (default: 100,000)
- `key_generator.default_ttl`: Key expiration time (default: 30s)
- `rag_system.max_results`: Maximum search results (default: 10)
