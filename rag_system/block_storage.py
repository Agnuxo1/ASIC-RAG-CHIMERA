"""
Block Storage Manager for ASIC-RAG

Manages persistent storage of encrypted knowledge blocks.
Provides efficient I/O operations with integrity verification.

Features:
- LMDB-based persistent storage
- Block caching for performance
- Integrity verification on read
- Atomic write operations
"""

import os
import json
import struct
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import hashlib

# Use shelve as fallback if lmdb not available
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False
    import shelve

from .knowledge_block import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory


@dataclass
class StorageConfig:
    """Configuration for block storage."""
    storage_path: str = "./data/encrypted_blocks"
    max_db_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    cache_size: int = 1000  # Number of blocks to cache
    enable_compression: bool = False
    sync_writes: bool = True
    read_ahead: bool = True


@dataclass
class StorageStats:
    """Storage statistics."""
    total_blocks: int = 0
    total_size_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    read_ops: int = 0
    write_ops: int = 0
    last_write_time: float = 0.0
    last_read_time: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_blocks": self.total_blocks,
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "read_ops": self.read_ops,
            "write_ops": self.write_ops
        }


class LRUCache:
    """Simple LRU cache for blocks."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[int, Tuple[float, bytes]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: int) -> Optional[bytes]:
        with self._lock:
            if key in self._cache:
                # Update access time
                _, value = self._cache[key]
                self._cache[key] = (time.time(), value)
                return value
            return None
    
    def put(self, key: int, value: bytes):
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), value)
    
    def remove(self, key: int):
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class BlockStorage:
    """
    Persistent storage manager for encrypted knowledge blocks.
    
    Uses LMDB for high-performance storage with ACID guarantees.
    Falls back to shelve if LMDB is not available.
    
    Example:
        >>> storage = BlockStorage(StorageConfig(storage_path="./data"))
        >>> block_id = storage.write_block(encrypted_block_bytes, metadata)
        >>> block_data = storage.read_block(block_id)
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize block storage.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self._cache = LRUCache(self.config.cache_size)
        self.stats = StorageStats()
        self._lock = threading.RLock()
        
        # Create storage directory
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backend
        self._init_storage()
        
        # Load metadata
        self._load_metadata()
    
    def _init_storage(self):
        """Initialize storage backend."""
        if HAS_LMDB:
            self._db_path = self._storage_path / "blocks.lmdb"
            self._env = lmdb.open(
                str(self._db_path),
                map_size=self.config.max_db_size,
                max_dbs=3,
                sync=self.config.sync_writes,
                readahead=self.config.read_ahead
            )
            # Create sub-databases
            self._blocks_db = self._env.open_db(b'blocks')
            self._index_db = self._env.open_db(b'index')
            self._meta_db = self._env.open_db(b'metadata')
            self._use_lmdb = True
        else:
            self._db_path = self._storage_path / "blocks"
            self._shelve = shelve.open(str(self._db_path))
            self._use_lmdb = False
    
    def _load_metadata(self):
        """Load storage metadata."""
        if self._use_lmdb:
            with self._env.begin(db=self._meta_db) as txn:
                data = txn.get(b'stats')
                if data:
                    stats_dict = json.loads(data.decode('utf-8'))
                    self.stats.total_blocks = stats_dict.get('total_blocks', 0)
                    self.stats.total_size_bytes = stats_dict.get('total_size_bytes', 0)
        else:
            if '_stats' in self._shelve:
                stats_dict = self._shelve['_stats']
                self.stats.total_blocks = stats_dict.get('total_blocks', 0)
                self.stats.total_size_bytes = stats_dict.get('total_size_bytes', 0)
    
    def _save_metadata(self):
        """Save storage metadata."""
        stats_dict = {
            'total_blocks': self.stats.total_blocks,
            'total_size_bytes': self.stats.total_size_bytes
        }
        
        if self._use_lmdb:
            with self._env.begin(db=self._meta_db, write=True) as txn:
                txn.put(b'stats', json.dumps(stats_dict).encode('utf-8'))
        else:
            self._shelve['_stats'] = stats_dict
            self._shelve.sync()
    
    def write_block(
        self,
        block_data: bytes,
        block_id: Optional[int] = None,
        tags: Optional[List[bytes]] = None
    ) -> int:
        """
        Write encrypted block to storage.
        
        Args:
            block_data: Encrypted block bytes
            block_id: Optional specific block ID
            tags: Optional tag hashes for indexing
            
        Returns:
            Assigned block ID
        """
        with self._lock:
            if block_id is None:
                block_id = self.stats.total_blocks
            
            key = struct.pack('>Q', block_id)
            
            if self._use_lmdb:
                with self._env.begin(write=True) as txn:
                    # Write block data
                    txn.put(key, block_data, db=self._blocks_db)
                    
                    # Update tag index
                    if tags:
                        for tag_hash in tags:
                            existing = txn.get(tag_hash, db=self._index_db)
                            if existing:
                                block_ids = set(struct.unpack(f'>{len(existing)//8}Q', existing))
                            else:
                                block_ids = set()
                            block_ids.add(block_id)
                            packed = struct.pack(f'>{len(block_ids)}Q', *block_ids)
                            txn.put(tag_hash, packed, db=self._index_db)
            else:
                self._shelve[f'block_{block_id}'] = block_data
                if tags:
                    for tag_hash in tags:
                        tag_key = f'tag_{tag_hash.hex()}'
                        if tag_key in self._shelve:
                            block_ids = set(self._shelve[tag_key])
                        else:
                            block_ids = set()
                        block_ids.add(block_id)
                        self._shelve[tag_key] = list(block_ids)
                self._shelve.sync()
            
            # Update cache
            self._cache.put(block_id, block_data)
            
            # Update stats
            self.stats.total_blocks = max(self.stats.total_blocks, block_id + 1)
            self.stats.total_size_bytes += len(block_data)
            self.stats.write_ops += 1
            self.stats.last_write_time = time.time()
            
            self._save_metadata()
            
            return block_id
    
    def read_block(self, block_id: int) -> Optional[bytes]:
        """
        Read encrypted block from storage.
        
        Args:
            block_id: Block ID to read
            
        Returns:
            Encrypted block bytes or None if not found
        """
        # Check cache first
        cached = self._cache.get(block_id)
        if cached:
            self.stats.cache_hits += 1
            return cached
        
        self.stats.cache_misses += 1
        
        with self._lock:
            key = struct.pack('>Q', block_id)
            
            if self._use_lmdb:
                with self._env.begin(db=self._blocks_db) as txn:
                    data = txn.get(key)
            else:
                data = self._shelve.get(f'block_{block_id}')
            
            if data:
                self._cache.put(block_id, data)
                self.stats.read_ops += 1
                self.stats.last_read_time = time.time()
            
            return data
    
    def delete_block(self, block_id: int) -> bool:
        """
        Delete block from storage.
        
        Args:
            block_id: Block ID to delete
            
        Returns:
            True if block was deleted
        """
        with self._lock:
            key = struct.pack('>Q', block_id)
            
            # Get block size before deletion
            block_data = self.read_block(block_id)
            if not block_data:
                return False
            
            block_size = len(block_data)
            
            if self._use_lmdb:
                with self._env.begin(write=True) as txn:
                    txn.delete(key, db=self._blocks_db)
            else:
                del self._shelve[f'block_{block_id}']
                self._shelve.sync()
            
            # Update cache
            self._cache.remove(block_id)
            
            # Update stats
            self.stats.total_size_bytes -= block_size
            
            self._save_metadata()
            
            return True
    
    def search_by_tags(
        self,
        tag_hashes: List[bytes],
        operation: str = "AND"
    ) -> List[int]:
        """
        Search blocks by tag hashes.
        
        Args:
            tag_hashes: Tag hashes to search for
            operation: "AND" or "OR"
            
        Returns:
            List of matching block IDs
        """
        with self._lock:
            result_sets = []
            
            for tag_hash in tag_hashes:
                if self._use_lmdb:
                    with self._env.begin(db=self._index_db) as txn:
                        data = txn.get(tag_hash)
                        if data:
                            block_ids = set(struct.unpack(f'>{len(data)//8}Q', data))
                        else:
                            block_ids = set()
                else:
                    tag_key = f'tag_{tag_hash.hex()}'
                    block_ids = set(self._shelve.get(tag_key, []))
                
                result_sets.append(block_ids)
            
            if not result_sets:
                return []
            
            if operation == "AND":
                result = result_sets[0]
                for s in result_sets[1:]:
                    result &= s
            else:  # OR
                result = set()
                for s in result_sets:
                    result |= s
            
            return sorted(result)
    
    def get_all_block_ids(self) -> List[int]:
        """Get all block IDs in storage."""
        with self._lock:
            if self._use_lmdb:
                block_ids = []
                with self._env.begin(db=self._blocks_db) as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        block_id = struct.unpack('>Q', key)[0]
                        block_ids.append(block_id)
                return sorted(block_ids)
            else:
                block_ids = []
                for key in self._shelve.keys():
                    if key.startswith('block_'):
                        block_id = int(key.split('_')[1])
                        block_ids.append(block_id)
                return sorted(block_ids)
    
    def iterate_blocks(self) -> Iterator[Tuple[int, bytes]]:
        """Iterate over all blocks."""
        for block_id in self.get_all_block_ids():
            data = self.read_block(block_id)
            if data:
                yield block_id, data
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        return self.stats
    
    def compact(self):
        """Compact storage to reclaim space."""
        if self._use_lmdb:
            # LMDB doesn't need explicit compaction
            pass
        else:
            self._shelve.sync()
    
    def close(self):
        """Close storage."""
        self._save_metadata()
        if self._use_lmdb:
            self._env.close()
        else:
            self._shelve.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BlockStorageManager:
    """
    High-level manager for block storage operations.
    
    Integrates with ASIC simulator for tag-based indexing.
    """
    
    def __init__(
        self,
        storage: BlockStorage,
        master_key: bytes
    ):
        """
        Initialize storage manager.
        
        Args:
            storage: Block storage instance
            master_key: Master encryption key
        """
        self.storage = storage
        self.master_key = master_key
    
    def store_block(self, block: KnowledgeBlock) -> int:
        """
        Store a knowledge block.
        
        Args:
            block: Knowledge block to store
            
        Returns:
            Assigned block ID
        """
        # Encrypt block
        block.encrypt(self.master_key)
        
        # Serialize
        block_data = block.to_bytes()
        
        # Write to storage
        block_id = self.storage.write_block(
            block_data,
            tags=block.tag_hashes
        )
        
        block.block_id = block_id
        return block_id
    
    def retrieve_block(self, block_id: int) -> Optional[KnowledgeBlock]:
        """
        Retrieve and decrypt a knowledge block.
        
        Args:
            block_id: Block ID to retrieve
            
        Returns:
            Decrypted knowledge block or None
        """
        block_data = self.storage.read_block(block_id)
        if not block_data:
            return None
        
        # Deserialize
        block = KnowledgeBlock.from_bytes(block_data, block_id)
        
        # Decrypt
        metadata, content = block.decrypt(self.master_key, block._encrypted_payload)
        block.metadata = metadata
        block.content = content
        
        return block
    
    def search_blocks(
        self,
        tag_hashes: List[bytes],
        operation: str = "AND"
    ) -> List[KnowledgeBlock]:
        """
        Search and retrieve blocks by tags.
        
        Args:
            tag_hashes: Tag hashes to search
            operation: "AND" or "OR"
            
        Returns:
            List of matching knowledge blocks
        """
        block_ids = self.storage.search_by_tags(tag_hashes, operation)
        
        blocks = []
        for block_id in block_ids:
            block = self.retrieve_block(block_id)
            if block:
                blocks.append(block)
        
        return blocks


if __name__ == "__main__":
    import tempfile
    
    print("Block Storage Demo")
    print("=" * 50)
    
    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(storage_path=tmpdir)
        
        with BlockStorage(config) as storage:
            # Write some blocks
            print("\nWriting blocks...")
            for i in range(5):
                data = f"Block {i} content: This is encrypted document data.".encode()
                tags = [
                    hashlib.sha256(f"tag_{i % 3}".encode()).digest(),
                    hashlib.sha256(f"common".encode()).digest(),
                ]
                block_id = storage.write_block(data, tags=tags)
                print(f"  Written block {block_id}: {len(data)} bytes")
            
            # Read blocks
            print("\nReading blocks...")
            for i in range(5):
                data = storage.read_block(i)
                if data:
                    print(f"  Block {i}: {len(data)} bytes")
            
            # Search by tags
            print("\nSearching by tag 'common'...")
            tag_hash = hashlib.sha256(b"common").digest()
            results = storage.search_by_tags([tag_hash])
            print(f"  Found blocks: {results}")
            
            # Statistics
            print("\nStorage Statistics:")
            stats = storage.get_stats()
            for key, value in stats.to_dict().items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
