"""
Texture Memory Manager for CHIMERA Integration

Implements GPU texture memory interface for:
- Fast index storage
- Efficient read patterns
- Cache-optimized access

This follows the CHIMERA philosophy of using GPU graphics
primitives for general-purpose computation.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import struct

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TextureBlock:
    """
    A block of data stored in GPU texture memory.
    
    Attributes:
        block_id: Unique block identifier
        data: Binary data content
        width: Texture width in pixels
        height: Texture height in pixels
        format: Data format (RGBA8, R32F, etc.)
        gpu_handle: GPU memory handle
    """
    block_id: int
    data: bytes
    width: int = 0
    height: int = 0
    format: str = "RGBA8"
    gpu_handle: Optional[Any] = None
    
    def __post_init__(self):
        if self.width == 0 or self.height == 0:
            # Calculate dimensions to fit data
            data_size = len(self.data)
            pixel_size = 4  # RGBA8 = 4 bytes per pixel
            total_pixels = (data_size + pixel_size - 1) // pixel_size
            
            # Make it square-ish
            import math
            side = int(math.ceil(math.sqrt(total_pixels)))
            self.width = side
            self.height = side


@dataclass
class TextureAtlas:
    """
    Atlas containing multiple texture blocks.
    
    Efficiently packs multiple small blocks into
    larger textures for better GPU memory utilization.
    """
    atlas_id: int
    width: int
    height: int
    blocks: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)  # block_id -> (x, y, w, h)
    data: Optional[bytes] = None
    gpu_handle: Optional[Any] = None


class TextureMemoryManager:
    """
    GPU texture memory manager for CHIMERA.
    
    Provides efficient storage and retrieval of data
    using GPU texture memory, which offers:
    - Hardware texture caching
    - Efficient 2D access patterns
    - Fast parallel reads
    
    Example:
        >>> manager = TextureMemoryManager()
        >>> block_id = manager.allocate(b"data to store")
        >>> data = manager.read(block_id)
    """
    
    def __init__(
        self,
        max_texture_size: int = 4096,
        use_gpu: bool = True
    ):
        """
        Initialize texture memory manager.
        
        Args:
            max_texture_size: Maximum texture dimension
            use_gpu: Use GPU memory if available
        """
        self.max_texture_size = max_texture_size
        self._use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()
        
        # Storage
        self._blocks: Dict[int, TextureBlock] = {}
        self._atlases: Dict[int, TextureAtlas] = {}
        self._next_block_id = 0
        self._next_atlas_id = 0
        
        # GPU tensors (if using GPU)
        self._gpu_textures: Dict[int, Any] = {}
        
        # Statistics
        self._total_allocated = 0
        self._total_reads = 0
        self._total_writes = 0
    
    @property
    def is_gpu_available(self) -> bool:
        return self._use_gpu
    
    def allocate(self, data: bytes) -> int:
        """
        Allocate texture memory for data.
        
        Args:
            data: Binary data to store
            
        Returns:
            Block ID for the allocated texture
        """
        block_id = self._next_block_id
        self._next_block_id += 1
        
        block = TextureBlock(
            block_id=block_id,
            data=data
        )
        
        # Upload to GPU if available
        if self._use_gpu:
            self._upload_to_gpu(block)
        
        self._blocks[block_id] = block
        self._total_allocated += len(data)
        self._total_writes += 1
        
        return block_id
    
    def read(self, block_id: int) -> Optional[bytes]:
        """
        Read data from texture memory.
        
        Args:
            block_id: Block ID to read
            
        Returns:
            Data bytes or None if not found
        """
        if block_id not in self._blocks:
            return None
        
        self._total_reads += 1
        
        block = self._blocks[block_id]
        
        if self._use_gpu and block.gpu_handle is not None:
            return self._download_from_gpu(block)
        
        return block.data
    
    def free(self, block_id: int) -> bool:
        """
        Free texture memory.
        
        Args:
            block_id: Block ID to free
            
        Returns:
            True if freed successfully
        """
        if block_id not in self._blocks:
            return False
        
        block = self._blocks[block_id]
        
        if self._use_gpu and block.gpu_handle is not None:
            self._free_gpu_memory(block)
        
        self._total_allocated -= len(block.data)
        del self._blocks[block_id]
        
        return True
    
    def _upload_to_gpu(self, block: TextureBlock):
        """Upload block data to GPU texture memory."""
        if not self._use_gpu:
            return
        
        # Convert data to numpy array
        data_array = np.frombuffer(
            block.data + b'\x00' * (block.width * block.height * 4 - len(block.data)),
            dtype=np.uint8
        )
        
        # Reshape to texture dimensions
        data_array = data_array.reshape(block.height, block.width, 4)
        
        # Upload to GPU
        tensor = torch.from_numpy(data_array).to('cuda')
        block.gpu_handle = tensor
        self._gpu_textures[block.block_id] = tensor
    
    def _download_from_gpu(self, block: TextureBlock) -> bytes:
        """Download block data from GPU."""
        if block.gpu_handle is None:
            return block.data
        
        tensor = block.gpu_handle
        data_array = tensor.cpu().numpy()
        
        # Flatten and convert to bytes
        flat = data_array.flatten()
        
        # Return only the original data length
        return bytes(flat[:len(block.data)])
    
    def _free_gpu_memory(self, block: TextureBlock):
        """Free GPU memory for block."""
        if block.block_id in self._gpu_textures:
            del self._gpu_textures[block.block_id]
        block.gpu_handle = None
    
    def create_index_texture(
        self,
        index_data: Dict[bytes, List[int]]
    ) -> int:
        """
        Create a texture storing tag index data.
        
        Args:
            index_data: Mapping of tag hashes to block IDs
            
        Returns:
            Atlas ID for the index texture
        """
        # Serialize index data
        entries = []
        for tag_hash, block_ids in index_data.items():
            # Format: tag_hash (32 bytes) + count (4 bytes) + block_ids (4 bytes each)
            entry = tag_hash + struct.pack('>I', len(block_ids))
            for bid in block_ids:
                entry += struct.pack('>I', bid)
            entries.append(entry)
        
        # Pack into texture
        packed_data = b''.join(entries)
        
        atlas_id = self._next_atlas_id
        self._next_atlas_id += 1
        
        atlas = TextureAtlas(
            atlas_id=atlas_id,
            width=self.max_texture_size,
            height=(len(packed_data) + self.max_texture_size * 4 - 1) // (self.max_texture_size * 4),
            data=packed_data
        )
        
        self._atlases[atlas_id] = atlas
        
        return atlas_id
    
    def lookup_index(
        self,
        atlas_id: int,
        tag_hash: bytes
    ) -> List[int]:
        """
        Lookup block IDs for a tag in index texture.
        
        Args:
            atlas_id: Atlas containing the index
            tag_hash: Tag hash to lookup
            
        Returns:
            List of block IDs
        """
        if atlas_id not in self._atlases:
            return []
        
        atlas = self._atlases[atlas_id]
        data = atlas.data
        
        # Linear search through packed data
        # In production, use GPU-based binary search
        offset = 0
        while offset < len(data):
            stored_hash = data[offset:offset + 32]
            offset += 32
            
            if offset + 4 > len(data):
                break
            
            count = struct.unpack('>I', data[offset:offset + 4])[0]
            offset += 4
            
            if stored_hash == tag_hash:
                block_ids = []
                for _ in range(count):
                    if offset + 4 > len(data):
                        break
                    bid = struct.unpack('>I', data[offset:offset + 4])[0]
                    block_ids.append(bid)
                    offset += 4
                return block_ids
            else:
                offset += count * 4
        
        return []
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics."""
        gpu_memory = 0
        if self._use_gpu:
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return {
            "total_blocks": len(self._blocks),
            "total_atlases": len(self._atlases),
            "total_allocated_bytes": self._total_allocated,
            "total_allocated_mb": self._total_allocated / (1024 * 1024),
            "gpu_memory_mb": gpu_memory,
            "total_reads": self._total_reads,
            "total_writes": self._total_writes
        }
    
    def clear(self):
        """Clear all texture memory."""
        for block_id in list(self._blocks.keys()):
            self.free(block_id)
        
        self._blocks.clear()
        self._atlases.clear()
        self._gpu_textures.clear()
        
        if self._use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Texture Memory Manager Demo")
    print("=" * 50)
    
    manager = TextureMemoryManager()
    
    print(f"\nGPU Available: {manager.is_gpu_available}")
    
    # Allocate some blocks
    print("\nAllocating blocks...")
    block_ids = []
    for i in range(10):
        data = f"Block {i} content: " + "x" * 1000
        block_id = manager.allocate(data.encode())
        block_ids.append(block_id)
        print(f"  Allocated block {block_id}")
    
    # Read back
    print("\nReading blocks...")
    for block_id in block_ids[:3]:
        data = manager.read(block_id)
        print(f"  Block {block_id}: {len(data)} bytes")
    
    # Create index texture
    print("\nCreating index texture...")
    import hashlib
    index_data = {
        hashlib.sha256(b"finance").digest(): [0, 1, 2],
        hashlib.sha256(b"legal").digest(): [3, 4],
        hashlib.sha256(b"technical").digest(): [5, 6, 7, 8, 9],
    }
    atlas_id = manager.create_index_texture(index_data)
    print(f"  Created atlas {atlas_id}")
    
    # Lookup in index
    print("\nLooking up tags...")
    finance_hash = hashlib.sha256(b"finance").digest()
    results = manager.lookup_index(atlas_id, finance_hash)
    print(f"  'finance' -> blocks {results}")
    
    # Memory usage
    print("\nMemory Usage:")
    usage = manager.get_memory_usage()
    for key, value in usage.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup
    manager.clear()
