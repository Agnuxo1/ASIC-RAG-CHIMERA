"""
Knowledge Block Definition for ASIC-RAG

Core data structure for the system. Represents a unit of knowledge
that can be encrypted, stored, and retrieved.
"""

import time
import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import secrets

class BlockCategory(Enum):
    """Category of knowledge block."""
    GENERAL = "general"
    FINANCE = "finance"
    LEGAL = "legal"
    TECHNICAL = "technical"
    HR = "hr"
    MARKETING = "marketing"
    RESEARCH = "research"

@dataclass
class BlockMetadata:
    """Metadata for a knowledge block."""
    category: BlockCategory = BlockCategory.GENERAL
    source: str = "unknown"
    author: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "source": self.source,
            "author": self.author,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "custom": self.custom
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BlockMetadata":
        return cls(
            category=BlockCategory(data.get("category", "general")),
            source=data.get("source", "unknown"),
            author=data.get("author"),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time()),
            custom=data.get("custom", {})
        )

@dataclass
class BlockHeader:
    """Header for blockchain-like linking."""
    prev_hash: bytes
    timestamp: float
    nonce: int
    block_hash: bytes = field(init=False)
    
    def __post_init__(self):
        self.block_hash = self.compute_hash()
        
    def compute_hash(self) -> bytes:
        """Compute hash of header fields."""
        data = (
            self.prev_hash +
            struct.pack(">d", self.timestamp) +
            struct.pack(">Q", self.nonce)
        )
        return hashlib.sha256(data).digest()
    
    def to_bytes(self) -> bytes:
        """Serialize header."""
        return (
            self.prev_hash +
            struct.pack(">d", self.timestamp) +
            struct.pack(">Q", self.nonce)
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockHeader":
        """Deserialize header."""
        prev_hash = data[:32]
        timestamp = struct.unpack(">d", data[32:40])[0]
        nonce = struct.unpack(">Q", data[40:48])[0]
        return cls(prev_hash, timestamp, nonce)

@dataclass
class KnowledgeBlock:
    """
    Encrypted unit of knowledge.
    
    Contains:
    - Header (for integrity/linking)
    - Metadata (unencrypted for indexing)
    - Content (encrypted)
    - Tag Hashes (for blind indexing)
    """
    block_id: int
    header: BlockHeader
    tag_hashes: List[bytes]
    metadata: BlockMetadata
    content: bytes  # Can be plaintext or ciphertext
    
    _is_encrypted: bool = False
    _encrypted_payload: Optional[bytes] = None
    
    def encrypt(self, key: bytes):
        """Encrypt content using AES-GCM."""
        if self._is_encrypted:
            return
            
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        
        # Serialize content and metadata for encryption
        # We encrypt metadata too to ensure confidentiality, 
        # but keep a copy for indexing if needed (policy dependent)
        payload = json.dumps({
            "content": self.content.decode('utf-8', errors='ignore') if isinstance(self.content, bytes) else self.content,
            "metadata": self.metadata.to_dict()
        }).encode('utf-8')
        
        ciphertext = aesgcm.encrypt(nonce, payload, None)
        
        self._encrypted_payload = nonce + ciphertext
        self.content = b"" # Clear plaintext
        self._is_encrypted = True
        
    def decrypt(self, key: bytes, payload: Optional[bytes] = None) -> Tuple[BlockMetadata, bytes]:
        """Decrypt content."""
        if not self._is_encrypted and payload is None:
            return self.metadata, self.content
            
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        data = payload or self._encrypted_payload
        if not data:
            raise ValueError("No encrypted data found")
            
        nonce = data[:12]
        ciphertext = data[12:]
        
        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            data_dict = json.loads(plaintext.decode('utf-8'))
            
            content = data_dict["content"].encode('utf-8')
            metadata = BlockMetadata.from_dict(data_dict["metadata"])
            
            return metadata, content
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def verify_integrity(self) -> bool:
        """
        Verify the integrity of this block.
        
        Checks that:
        1. The header hash matches the computed hash
        2. Content has not been tampered with (via content hash comparison)
        
        Returns:
            True if block integrity is valid, False otherwise
        """
        # Compute expected header hash
        expected_hash = self.header.compute_hash()
        
        if expected_hash != self.header.block_hash:
            return False
        
        # Store original content hash for comparison
        if not hasattr(self, '_original_content_hash'):
            # First call - store the hash
            if self._is_encrypted:
                self._original_content_hash = hashlib.sha256(
                    self._encrypted_payload or b''
                ).digest()
            else:
                self._original_content_hash = hashlib.sha256(self.content).digest()
            return True
        
        # Subsequent calls - compare with stored hash
        if self._is_encrypted:
            current_hash = hashlib.sha256(self._encrypted_payload or b'').digest()
        else:
            current_hash = hashlib.sha256(self.content).digest()
        
        return current_hash == self._original_content_hash

    def to_bytes(self) -> bytes:
        """Serialize entire block."""
        # Format:
        # Header (48 bytes)
        # Num Tags (4 bytes)
        # Tags (32 * Num Tags bytes)
        # Metadata Len (4 bytes)
        # Metadata JSON (N bytes) - Unencrypted part for basic routing
        # Payload Len (4 bytes)
        # Payload (M bytes) - Encrypted content + full metadata
        
        header_bytes = self.header.to_bytes()
        
        tags_bytes = b"".join(self.tag_hashes)
        
        # We store minimal metadata unencrypted for routing/stats
        # Full metadata is inside encrypted payload
        meta_json = json.dumps({
            "category": self.metadata.category.value,
            "source": self.metadata.source,
            "created_at": self.metadata.created_at
        }).encode('utf-8')
        
        payload = self._encrypted_payload if self._is_encrypted else self.content
        
        return (
            header_bytes +
            struct.pack(">I", len(self.tag_hashes)) +
            tags_bytes +
            struct.pack(">I", len(meta_json)) +
            meta_json +
            struct.pack(">I", len(payload)) +
            payload
        )

    @classmethod
    def from_bytes(cls, data: bytes, block_id: int = 0) -> "KnowledgeBlock":
        """Deserialize block."""
        offset = 0
        
        # Header
        header = BlockHeader.from_bytes(data[offset:offset+48])
        offset += 48
        
        # Tags
        num_tags = struct.unpack(">I", data[offset:offset+4])[0]
        offset += 4
        
        tag_hashes = []
        for _ in range(num_tags):
            tag_hashes.append(data[offset:offset+32])
            offset += 32
            
        # Metadata (Unencrypted subset)
        meta_len = struct.unpack(">I", data[offset:offset+4])[0]
        offset += 4
        
        meta_json = data[offset:offset+meta_len]
        meta_dict = json.loads(meta_json.decode('utf-8'))
        offset += meta_len
        
        # Reconstruct partial metadata
        metadata = BlockMetadata(
            category=BlockCategory(meta_dict.get("category", "general")),
            source=meta_dict.get("source", "unknown"),
            created_at=meta_dict.get("created_at", 0.0)
        )
        
        # Payload
        payload_len = struct.unpack(">I", data[offset:offset+4])[0]
        offset += 4
        
        payload = data[offset:offset+payload_len]
        
        return cls(
            block_id=block_id,
            header=header,
            tag_hashes=tag_hashes,
            metadata=metadata,
            content=b"", # Content is in payload
            _is_encrypted=True,
            _encrypted_payload=payload
        )


class BlockChain:
    """
    A chain of KnowledgeBlocks with integrity verification.
    
    Maintains blocks in a linked list structure where each block
    references the hash of the previous block.
    """
    
    def __init__(self):
        """Initialize an empty blockchain."""
        self.blocks: List[KnowledgeBlock] = []
        self._genesis_hash = b'\x00' * 32
    
    def add_block(self, block: KnowledgeBlock):
        """
        Add a block to the chain.
        
        Args:
            block: KnowledgeBlock to add
        """
        self.blocks.append(block)
    
    def get_latest_hash(self) -> bytes:
        """
        Get the hash of the latest block.
        
        Returns:
            Hash of the latest block, or genesis hash if chain is empty
        """
        if not self.blocks:
            return self._genesis_hash
        return self.blocks[-1].header.block_hash
    
    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire chain.
        
        Checks that:
        1. Each block's prev_hash matches the previous block's block_hash
        2. Each block's header hash is valid
        
        Returns:
            True if chain is valid, False otherwise
        """
        if not self.blocks:
            return True
        
        # Check first block links to genesis
        if self.blocks[0].header.prev_hash != self._genesis_hash:
            return False
        
        # Verify each block's header hash
        for i, block in enumerate(self.blocks):
            expected_hash = block.header.compute_hash()
            if expected_hash != block.header.block_hash:
                return False
            
            # Check chain linkage (except for first block)
            if i > 0:
                prev_block = self.blocks[i - 1]
                if block.header.prev_hash != prev_block.header.block_hash:
                    return False
        
        return True
    
    def get_block(self, block_id: int) -> Optional[KnowledgeBlock]:
        """
        Get a block by its ID.
        
        Args:
            block_id: ID of the block to retrieve
            
        Returns:
            KnowledgeBlock if found, None otherwise
        """
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None
    
    def __len__(self) -> int:
        """Return the number of blocks in the chain."""
        return len(self.blocks)
