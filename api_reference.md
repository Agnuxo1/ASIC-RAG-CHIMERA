# ASIC-RAG-CHIMERA API Reference

## Table of Contents

1. [ASIC Simulator](#asic-simulator)
2. [RAG System](#rag-system)
3. [CHIMERA Integration](#chimera-integration)
4. [LLM Interface](#llm-interface)

---

## ASIC Simulator

### SHA256Engine

High-performance SHA-256 hash engine simulating ASIC behavior.

```python
from asic_simulator import SHA256Engine

engine = SHA256Engine(num_lanes=256, cache_size=100000)
```

#### Methods

##### `hash(data: bytes) -> HashResult`

Compute SHA-256 hash of single input.

**Parameters:**
- `data`: Input bytes to hash

**Returns:** `HashResult` with:
- `input_data`: Original input
- `hash_bytes`: 32-byte hash
- `hash_hex`: Hexadecimal string
- `computation_time_us`: Time in microseconds

##### `hash_batch(data_list: List[bytes], parallel: bool = True) -> HashBatch`

Compute hashes for multiple inputs in parallel.

**Parameters:**
- `data_list`: List of inputs to hash
- `parallel`: Use parallel processing (default: True)

**Returns:** `HashBatch` with:
- `results`: List of HashResult
- `total_time_us`: Total time
- `hashes_per_second`: Throughput metric

##### `double_hash(data: bytes) -> bytes`

Bitcoin-style double SHA-256: SHA256(SHA256(data)).

##### `verify_hash(data: bytes, expected_hash: bytes) -> bool`

Verify that data produces expected hash.

---

### MerkleTree

Hierarchical cryptographic tree for document verification.

```python
from asic_simulator import MerkleTree

tree = MerkleTree()
```

#### Methods

##### `add_leaf(data: bytes) -> int`

Add a leaf node to the tree.

**Returns:** Index of added leaf

##### `build() -> None`

Construct the tree from added leaves.

##### `get_proof(leaf_index: int) -> MerkleProof`

Generate inclusion proof for a leaf.

**Returns:** `MerkleProof` with:
- `leaf_hash`: Hash of the leaf
- `proof_path`: List of (hash, direction) tuples
- `root_hash`: Current root hash

##### `verify_proof(proof: MerkleProof) -> bool`

Verify a Merkle proof.

---

### IndexManager

Tag-based document indexing with boolean search.

```python
from asic_simulator import IndexManager, SearchOperation

index = IndexManager()
```

#### Methods

##### `add_tag(tag: bytes, block_id: int, category: str = None) -> None`

Add tag-to-block mapping.

##### `search(tags: List[bytes], operation: SearchOperation, limit: int = 100) -> SearchResult`

Search for blocks matching tags.

**Parameters:**
- `tags`: List of tag hashes to search
- `operation`: `SearchOperation.AND`, `OR`, or `NOT`
- `limit`: Maximum results

**Returns:** `SearchResult` with:
- `block_ids`: Matching block IDs
- `relevance_scores`: Score per block
- `search_time_us`: Query time

---

### KeyGenerator

Temporary key generation with automatic expiration.

```python
from asic_simulator import KeyGenerator

generator = KeyGenerator(master_key=key, default_key_ttl=30)
```

#### Methods

##### `create_session(ttl: float = None) -> KeySession`

Create a new key session.

##### `generate_key(session_id: str, block_hash: bytes) -> TemporaryKey`

Generate a temporary key for a block.

**Returns:** `TemporaryKey` with:
- `key_id`: Unique key identifier
- `key_bytes`: 32-byte AES key
- `expires_at`: Expiration timestamp

##### `use_key(key_id: str) -> Optional[bytes]`

Use a key if still valid.

**Returns:** Key bytes or None if expired/revoked

---

## RAG System

### KnowledgeBlock

Encrypted document storage block.

```python
from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory

block = KnowledgeBlock(
    block_id=1,
    header=header,
    tag_hashes=[...],
    metadata=metadata,
    content=b"document content"
)
```

#### Methods

##### `encrypt(master_key: bytes) -> None`

Encrypt block content with AES-256-GCM.

##### `decrypt(master_key: bytes) -> Tuple[BlockMetadata, bytes]`

Decrypt block and return metadata and content.

##### `to_bytes() -> bytes`

Serialize block to bytes.

##### `from_bytes(data: bytes) -> KnowledgeBlock`

Deserialize block from bytes.

---

### BlockStorage

Persistent encrypted block storage.

```python
from rag_system.block_storage import BlockStorage, StorageConfig

config = StorageConfig(storage_path="/path/to/storage")
storage = BlockStorage(config)
```

#### Methods

##### `write_block(data: bytes, tags: List[bytes] = None) -> int`

Write block to storage.

**Returns:** Assigned block ID

##### `read_block(block_id: int) -> Optional[bytes]`

Read block from storage.

##### `search_by_tags(tags: List[bytes], operation: str = "AND") -> List[int]`

Search blocks by tags.

##### `delete_block(block_id: int) -> bool`

Delete a block.

---

### DocumentProcessor

Document ingestion and processing pipeline.

```python
from rag_system import DocumentProcessor

processor = DocumentProcessor()
```

#### Methods

##### `process_text(text: str, source: str, category: BlockCategory = GENERAL) -> ProcessedDocument`

Process raw text content.

**Returns:** `ProcessedDocument` with:
- `content`: Original content
- `chunks`: Text chunks
- `keywords`: Extracted keywords
- `metadata`: Document metadata

##### `process_file(file_path: str) -> ProcessedDocument`

Process a document file.

##### `create_blocks(doc: ProcessedDocument) -> List[KnowledgeBlock]`

Create storage blocks from processed document.

---

### QueryEngine

RAG query processing engine.

```python
from rag_system import QueryEngine

engine = QueryEngine(storage_manager, index_manager)
```

#### Methods

##### `query(query_text: str, category: str = None, generate_answer: bool = True) -> QueryResult`

Execute a RAG query.

**Returns:** `QueryResult` with:
- `query`: Original query
- `retrieved_blocks`: List of RetrievedBlock
- `context`: Assembled context
- `answer`: Generated answer (if LLM available)

---

## CHIMERA Integration

### GPUHashEngine

GPU-accelerated hash computation.

```python
from chimera_integration import GPUHashEngine

engine = GPUHashEngine(device="auto")
```

#### Properties

- `is_gpu_available`: Whether CUDA is available
- `device`: Current device (cuda/cpu)

#### Methods

##### `hash_batch(data_list: List[bytes]) -> GPUHashBatch`

Batch hash computation on GPU.

---

### TextureMemoryManager

GPU texture memory for index storage.

```python
from chimera_integration import TextureMemoryManager

manager = TextureMemoryManager(max_texture_size=4096)
```

#### Methods

##### `allocate(data: bytes) -> int`

Allocate texture memory.

**Returns:** Block ID

##### `read(block_id: int) -> bytes`

Read from texture memory.

---

## LLM Interface

### QwenLoader

Qwen3-0.6B model loader.

```python
from llm_interface import QwenLoader, ModelConfig

config = ModelConfig(model_name="Qwen/Qwen3-0.6B")
loader = QwenLoader(config)
```

#### Methods

##### `load() -> bool`

Load model into memory.

##### `generate(prompt: str, max_new_tokens: int = 512) -> str`

Generate text completion.

##### `chat(messages: List[Dict], system_prompt: str = None) -> str`

Chat completion with message history.

---

### ResponseGenerator

RAG-aware response generation.

```python
from llm_interface import ResponseGenerator

generator = ResponseGenerator(llm_loader)
```

#### Methods

##### `generate(query: str, context: str, sources: List = None) -> GeneratedResponse`

Generate response for query with context.

**Returns:** `GeneratedResponse` with:
- `answer`: Generated answer
- `sources_used`: Referenced sources
- `confidence`: Confidence score
- `grounded`: Whether grounded in context

---

## Configuration

### YAML Configuration

```yaml
asic_simulator:
  num_lanes: 256
  batch_size: 1024
  cache_size: 100000

encryption:
  algorithm: "AES-256-GCM"
  pbkdf2_iterations: 100000
  nonce_size: 12

rag_system:
  max_block_size: 1048576
  max_results: 10
  relevance_threshold: 0.5

llm:
  model_name: "Qwen/Qwen3-0.6B"
  max_context_length: 8192
  use_4bit: true
```

---

## Error Handling

All modules raise standard Python exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing files
- `PermissionError`: Access denied
- `RuntimeError`: Operation failed

Encryption-specific:
- `cryptography.exceptions.InvalidTag`: Decryption failed (wrong key or corrupted data)

---

## Thread Safety

- `SHA256Engine`: Thread-safe with internal locking
- `IndexManager`: Thread-safe for reads, lock for writes
- `BlockStorage`: Thread-safe via LMDB transactions
- `KeyGenerator`: Thread-safe with cleanup thread

---

## Memory Management

For large datasets:
- Use streaming for document processing
- Close storage connections when done
- Clear GPU cache with `engine.clear_cache()`
- Unload LLM with `loader.unload()`
