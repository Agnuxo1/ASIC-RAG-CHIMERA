"""
Index Manager for ASIC-RAG

Manages the tag-based index for fast retrieval.
Simulates ASIC-accelerated tag matching.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Union
from enum import Enum
import threading

class SearchOperation(Enum):
    """Search set operations."""
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class SearchResult:
    """Result of an index search."""
    block_ids: List[int]
    relevance_scores: Dict[int, float]
    total_candidates: int
    search_time_ms: float

class IndexManager:
    """
    Manages tag-to-block index.
    
    Supports:
    - Fast tag lookup
    - Boolean set operations
    - Category filtering
    """
    
    def __init__(self):
        # Main index: tag_hash -> Set[block_id]
        self._index: Dict[bytes, Set[int]] = {}
        
        # Category index: category -> Set[block_id]
        self._category_index: Dict[str, Set[int]] = {}
        
        # Block tags: block_id -> Set[tag_hash] (for similarity search)
        self._block_tags: Dict[int, Set[bytes]] = {}
        
        self._lock = threading.RLock()
        
        # Stats
        self._total_tags = 0
        self._total_blocks = 0
        
    def add_tag(self, tag_hash: bytes, block_id: int, category: Optional[str] = None):
        """Add a tag to the index."""
        with self._lock:
            if tag_hash not in self._index:
                self._index[tag_hash] = set()
                self._total_tags += 1
            
            self._index[tag_hash].add(block_id)
            
            if block_id not in self._block_tags:
                self._block_tags[block_id] = set()
                self._total_blocks += 1
            self._block_tags[block_id].add(tag_hash)
            
            if category:
                if category not in self._category_index:
                    self._category_index[category] = set()
                self._category_index[category].add(block_id)
    
    def add_tags(self, tag_hashes: List[bytes], block_id: int, category: Optional[str] = None):
        """Add multiple tags to the index for a single block."""
        for tag_hash in tag_hashes:
            self.add_tag(tag_hash, block_id, category)
    
    def remove_tag(self, tag_hash: bytes, block_id: int):
        """Remove a tag association from a block."""
        with self._lock:
            if tag_hash in self._index:
                self._index[tag_hash].discard(block_id)
                if not self._index[tag_hash]:
                    del self._index[tag_hash]
                    self._total_tags -= 1
            
            if block_id in self._block_tags:
                self._block_tags[block_id].discard(tag_hash)

    def get_blocks_for_tag(self, tag_hash: bytes) -> Set[int]:
        """Get all blocks for a specific tag."""
        with self._lock:
            return self._index.get(tag_hash, set()).copy()
            
    def get_tags_for_block(self, block_id: int) -> List[bytes]:
        """Get all tags for a specific block."""
        with self._lock:
            return list(self._block_tags.get(block_id, set()))

    def search(
        self,
        tags: List[bytes],
        operation: Union[SearchOperation, str] = SearchOperation.OR,
        limit: int = 10,
        min_relevance: float = 0.0,
        category: Optional[str] = None
    ) -> SearchResult:
        """
        Search for blocks matching tags.
        
        Args:
            tags: List of tag hashes
            operation: Search operation (AND/OR)
            limit: Max results
            min_relevance: Minimum score
            category: Filter by category
            
        Returns:
            SearchResult
        """
        start_time = time.perf_counter()
        
        if isinstance(operation, str):
            operation = SearchOperation(operation.lower())
            
        with self._lock:
            result_sets = []
            for tag in tags:
                result_sets.append(self._index.get(tag, set()))
            
            if not result_sets:
                return SearchResult([], {}, 0, 0.0)
            
            # Apply set operation
            if operation == SearchOperation.AND:
                candidates = set.intersection(*result_sets) if result_sets else set()
            else: # OR
                candidates = set.union(*result_sets) if result_sets else set()
                
            # Apply category filter
            if category and category in self._category_index:
                candidates &= self._category_index[category]
            elif category:
                candidates = set() # Category exists but no blocks or invalid category
                
            # Calculate relevance scores
            # Simple scoring: count of matched tags / total query tags
            scores = {}
            for block_id in candidates:
                matched_count = 0
                block_tags = self._block_tags.get(block_id, set())
                
                for tag in tags:
                    if tag in block_tags:
                        matched_count += 1
                
                score = matched_count / len(tags)
                if score >= min_relevance:
                    scores[block_id] = score
            
            # Sort by score
            sorted_blocks = sorted(
                scores.keys(),
                key=lambda bid: scores[bid],
                reverse=True
            )
            
            final_blocks = sorted_blocks[:limit]
            
            end_time = time.perf_counter()
            
            return SearchResult(
                block_ids=final_blocks,
                relevance_scores={bid: scores[bid] for bid in final_blocks},
                total_candidates=len(candidates),
                search_time_ms=(end_time - start_time) * 1000
            )

    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            "total_tags": self._total_tags,
            "total_blocks": self._total_blocks,
            "unique_tags": len(self._index),
            "categories": list(self._category_index.keys())
        }
