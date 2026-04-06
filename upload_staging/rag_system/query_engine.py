"""
Query Engine for ASIC-RAG

Handles the complete RAG query pipeline:
- Query parsing and keyword extraction
- ASIC-accelerated tag search
- Block retrieval and decryption
- Context assembly for LLM
- Response generation
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum

from .knowledge_block import KnowledgeBlock, BlockMetadata
from .block_storage import BlockStorage, BlockStorageManager
from .document_processor import KeywordExtractor


class QueryMode(Enum):
    """Query processing modes."""
    KEYWORD = "keyword"  # Exact keyword matching
    SEMANTIC = "semantic"  # Semantic similarity (requires embeddings)
    HYBRID = "hybrid"  # Combined keyword + semantic


@dataclass
class QueryConfig:
    """Configuration for query engine."""
    max_results: int = 10
    relevance_threshold: float = 0.5
    mode: QueryMode = QueryMode.KEYWORD
    include_metadata: bool = True
    dedup_threshold: float = 0.9  # Similarity threshold for deduplication
    timeout_seconds: float = 30.0


@dataclass
class RetrievedBlock:
    """A retrieved block with relevance information."""
    block_id: int
    content: str
    metadata: BlockMetadata
    relevance_score: float
    matched_tags: List[str]
    decryption_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "relevance_score": self.relevance_score,
            "matched_tags": self.matched_tags,
            "category": self.metadata.category.value,
            "source": self.metadata.source
        }


@dataclass
class QueryResult:
    """Result of a RAG query."""
    query: str
    retrieved_blocks: List[RetrievedBlock]
    context: str
    answer: Optional[str] = None
    search_time_ms: float = 0.0
    total_time_ms: float = 0.0
    keywords_used: List[str] = field(default_factory=list)
    blocks_searched: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "num_results": len(self.retrieved_blocks),
            "search_time_ms": self.search_time_ms,
            "total_time_ms": self.total_time_ms,
            "keywords_used": self.keywords_used,
            "blocks_searched": self.blocks_searched,
            "sources": [
                {"block_id": b.block_id, "source": b.metadata.source, "score": b.relevance_score}
                for b in self.retrieved_blocks
            ]
        }


class QueryEngine:
    """
    Main query engine for ASIC-RAG system.
    
    Orchestrates the complete retrieval pipeline:
    1. Parse query and extract keywords
    2. Convert keywords to tag hashes
    3. Search index via ASIC simulator
    4. Retrieve and decrypt matching blocks
    5. Assemble context for LLM
    6. Generate response (if LLM provided)
    
    Example:
        >>> engine = QueryEngine(storage_manager, asic_index)
        >>> result = engine.query("What was Q3 revenue?")
        >>> print(result.context)
    """
    
    def __init__(
        self,
        storage_manager: BlockStorageManager,
        index_manager: Any,  # IndexManager from asic_simulator
        config: Optional[QueryConfig] = None,
        llm_generator: Optional[Callable[[str, str], str]] = None
    ):
        """
        Initialize query engine.
        
        Args:
            storage_manager: Block storage manager
            index_manager: ASIC index manager
            config: Query configuration
            llm_generator: Optional LLM function for response generation
        """
        self.storage = storage_manager
        self.index = index_manager
        self.config = config or QueryConfig()
        self.llm_generator = llm_generator
        
        self.keyword_extractor = KeywordExtractor(
            min_length=2,
            max_keywords=20
        )
        
        # Statistics
        self._total_queries = 0
        self._total_query_time_ms = 0.0
        self._cache_hits = 0
    
    def query(
        self,
        query_text: str,
        category_filter: Optional[str] = None,
        generate_answer: bool = True
    ) -> QueryResult:
        """
        Execute a RAG query.
        
        Args:
            query_text: Natural language query
            category_filter: Optional category to filter results
            generate_answer: Whether to generate LLM answer
            
        Returns:
            QueryResult with retrieved context and optional answer
        """
        start_time = time.perf_counter()
        
        # Step 1: Extract keywords from query
        keywords = self._extract_query_keywords(query_text)
        
        # Step 2: Convert to tag hashes
        tag_hashes = [
            hashlib.sha256(kw.encode('utf-8')).digest()
            for kw in keywords
        ]
        
        # Step 3: Search index
        search_start = time.perf_counter()
        search_result = self.index.search(
            tags=tag_hashes,
            operation="OR",  # Use OR for better recall
            limit=self.config.max_results * 2,  # Get extra for filtering
            min_relevance=self.config.relevance_threshold,
            category=category_filter
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Step 4: Retrieve and decrypt blocks
        retrieved_blocks = self._retrieve_blocks(
            search_result.block_ids,
            search_result.relevance_scores,
            keywords
        )
        
        # Step 5: Rank and filter results
        ranked_blocks = self._rank_results(retrieved_blocks, query_text)
        final_blocks = ranked_blocks[:self.config.max_results]
        
        # Step 6: Assemble context
        context = self._assemble_context(final_blocks)
        
        # Step 7: Generate answer if LLM available
        answer = None
        if generate_answer and self.llm_generator and context:
            answer = self._generate_answer(query_text, context)
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        
        # Update statistics
        self._total_queries += 1
        self._total_query_time_ms += total_time
        
        return QueryResult(
            query=query_text,
            retrieved_blocks=final_blocks,
            context=context,
            answer=answer,
            search_time_ms=search_time,
            total_time_ms=total_time,
            keywords_used=keywords,
            blocks_searched=search_result.total_candidates
        )
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from query text."""
        # Use keyword extractor
        keywords = self.keyword_extractor.extract(query)
        
        # Also include original query words (for exact matches)
        query_words = [
            w.lower() for w in query.split()
            if len(w) >= 2
        ]
        
        # Combine and deduplicate
        all_keywords = list(dict.fromkeys(keywords + query_words))
        
        return all_keywords[:20]  # Limit to top 20
    
    def _retrieve_blocks(
        self,
        block_ids: List[int],
        relevance_scores: Dict[int, float],
        query_keywords: List[str]
    ) -> List[RetrievedBlock]:
        """Retrieve and decrypt blocks."""
        retrieved = []
        
        for block_id in block_ids:
            decrypt_start = time.perf_counter()
            
            try:
                block = self.storage.retrieve_block(block_id)
                if block is None:
                    continue
                
                decrypt_time = (time.perf_counter() - decrypt_start) * 1000
                
                # Find matched tags
                content_lower = block.content.decode('utf-8', errors='ignore').lower()
                matched = [kw for kw in query_keywords if kw in content_lower]
                
                retrieved.append(RetrievedBlock(
                    block_id=block_id,
                    content=block.content.decode('utf-8', errors='ignore'),
                    metadata=block.metadata,
                    relevance_score=relevance_scores.get(block_id, 0.0),
                    matched_tags=matched,
                    decryption_time_ms=decrypt_time
                ))
                
            except Exception as e:
                # Log error but continue
                print(f"Error retrieving block {block_id}: {e}")
        
        return retrieved
    
    def _rank_results(
        self,
        blocks: List[RetrievedBlock],
        query: str
    ) -> List[RetrievedBlock]:
        """Rank retrieved blocks by relevance."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        def score_block(block: RetrievedBlock) -> float:
            # Base score from index search
            score = block.relevance_score
            
            # Boost for matched tags
            score += len(block.matched_tags) * 0.1
            
            # Boost for query words in content
            content_lower = block.content.lower()
            word_matches = sum(1 for w in query_words if w in content_lower)
            score += word_matches * 0.05
            
            # Boost for recent documents
            if block.metadata.modified_at:
                age_days = (time.time() - block.metadata.modified_at) / 86400
                recency_boost = max(0, 0.1 - age_days * 0.001)
                score += recency_boost
            
            return score
        
        # Score and sort
        for block in blocks:
            block.relevance_score = score_block(block)
        
        blocks.sort(key=lambda b: b.relevance_score, reverse=True)
        
        return blocks
    
    def _assemble_context(
        self,
        blocks: List[RetrievedBlock],
        max_length: int = 4000
    ) -> str:
        """Assemble context from retrieved blocks."""
        if not blocks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, block in enumerate(blocks):
            # Format block as context
            if self.config.include_metadata:
                block_context = f"[Document {i+1}] (Source: {block.metadata.source}, Category: {block.metadata.category.value})\n{block.content}\n"
            else:
                block_context = f"[Document {i+1}]\n{block.content}\n"
            
            # Check length
            if current_length + len(block_context) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length
                if remaining > 100:
                    block_context = block_context[:remaining] + "..."
                    context_parts.append(block_context)
                break
            
            context_parts.append(block_context)
            current_length += len(block_context)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM."""
        if not self.llm_generator:
            return ""
        
        try:
            return self.llm_generator(query, context)
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def search_similar(
        self,
        block_id: int,
        max_results: int = 5
    ) -> List[RetrievedBlock]:
        """
        Find blocks similar to a given block.
        
        Args:
            block_id: Reference block ID
            max_results: Maximum results to return
            
        Returns:
            List of similar blocks
        """
        # Get reference block tags
        ref_tags = self.index.get_tags_for_block(block_id)
        
        if not ref_tags:
            return []
        
        # Search for blocks with similar tags
        search_result = self.index.search(
            tags=ref_tags,
            operation="OR",
            limit=max_results + 1  # +1 to exclude self
        )
        
        # Retrieve blocks (excluding reference)
        blocks = []
        for bid in search_result.block_ids:
            if bid != block_id:
                block = self.storage.retrieve_block(bid)
                if block:
                    blocks.append(RetrievedBlock(
                        block_id=bid,
                        content=block.content.decode('utf-8', errors='ignore'),
                        metadata=block.metadata,
                        relevance_score=search_result.relevance_scores.get(bid, 0.0),
                        matched_tags=[],
                        decryption_time_ms=0.0
                    ))
        
        return blocks[:max_results]
    
    def get_statistics(self) -> Dict:
        """Get query engine statistics."""
        return {
            "total_queries": self._total_queries,
            "total_query_time_ms": self._total_query_time_ms,
            "average_query_time_ms": (
                self._total_query_time_ms / self._total_queries
                if self._total_queries > 0 else 0.0
            ),
            "cache_hits": self._cache_hits
        }


class RAGPipeline:
    """
    Complete RAG pipeline integrating all components.
    
    Provides a high-level interface for the ASIC-RAG system.
    """
    
    def __init__(
        self,
        storage_path: str,
        master_key: bytes,
        llm_generator: Optional[Callable[[str, str], str]] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            storage_path: Path to block storage
            master_key: Master encryption key
            llm_generator: Optional LLM function
        """
        from .block_storage import BlockStorage, StorageConfig, BlockStorageManager
        from asic_simulator import IndexManager
        
        # Initialize components
        storage_config = StorageConfig(storage_path=storage_path)
        self.storage = BlockStorage(storage_config)
        self.storage_manager = BlockStorageManager(self.storage, master_key)
        self.index = IndexManager()
        self.query_engine = QueryEngine(
            self.storage_manager,
            self.index,
            llm_generator=llm_generator
        )
        
        from .document_processor import DocumentProcessor
        self.processor = DocumentProcessor()
        
        self.master_key = master_key
    
    def ingest(
        self,
        content: str,
        source: str = "direct_input",
        tags: Optional[List[str]] = None
    ) -> List[int]:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content
            source: Source identifier
            tags: Optional additional tags
            
        Returns:
            List of created block IDs
        """
        # Process document
        processed = self.processor.process_text(content, source)
        
        # Add custom tags
        if tags:
            processed.keywords.extend(tags)
        
        # Create blocks
        blocks = self.processor.create_blocks(processed)
        
        # Store blocks and update index
        block_ids = []
        for block in blocks:
            # Encrypt and store
            block.encrypt(self.master_key)
            block_id = self.storage.write_block(
                block.to_bytes(),
                tags=block.tag_hashes
            )
            block_ids.append(block_id)
            
            # Update index
            for tag_hash in block.tag_hashes:
                self.index.add_tag(tag_hash, block_id)
        
        return block_ids
    
    def query(self, question: str) -> QueryResult:
        """
        Query the RAG system.
        
        Args:
            question: Natural language question
            
        Returns:
            QueryResult with context and answer
        """
        return self.query_engine.query(question)
    
    def close(self):
        """Close the pipeline and release resources."""
        self.storage.close()


if __name__ == "__main__":
    print("Query Engine Demo")
    print("=" * 50)
    
    # Demo without actual storage (mock)
    class MockStorageManager:
        def retrieve_block(self, block_id):
            from .knowledge_block import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
            return type('Block', (), {
                'content': f"Mock content for block {block_id}. Contains financial data.".encode(),
                'metadata': BlockMetadata(category=BlockCategory.FINANCE, source=f"doc_{block_id}.txt")
            })()
    
    class MockIndex:
        def search(self, tags, operation, limit, min_relevance, category):
            return type('Result', (), {
                'block_ids': [0, 1, 2],
                'relevance_scores': {0: 0.9, 1: 0.7, 2: 0.5},
                'total_candidates': 3
            })()
        
        def get_tags_for_block(self, block_id):
            return []
    
    # Create mock engine
    config = QueryConfig(max_results=5)
    engine = QueryEngine(
        MockStorageManager(),
        MockIndex(),
        config=config
    )
    
    # Execute query
    result = engine.query("What was the Q3 revenue?")
    
    print(f"\nQuery: {result.query}")
    print(f"Keywords used: {result.keywords_used}")
    print(f"Blocks searched: {result.blocks_searched}")
    print(f"Results retrieved: {len(result.retrieved_blocks)}")
    print(f"Search time: {result.search_time_ms:.2f} ms")
    print(f"Total time: {result.total_time_ms:.2f} ms")
    
    print("\nContext preview:")
    print(result.context[:500] + "..." if len(result.context) > 500 else result.context)
