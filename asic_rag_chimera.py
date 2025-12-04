"""
ASIC-RAG-CHIMERA Main Integration Module

Provides unified interface for the complete system:
- Document ingestion
- Query processing
- System management

Usage:
    from asic_rag_chimera import ASICRAGSystem
    
    system = ASICRAGSystem(storage_path="./data", master_key=key)
    system.ingest("document.txt")
    result = system.query("What is the revenue?")
"""

import os
import yaml
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class SystemConfig:
    """Configuration for ASIC-RAG-CHIMERA system."""
    storage_path: str = "./data"
    config_path: Optional[str] = None
    
    # ASIC simulator settings
    num_lanes: int = 256
    batch_size: int = 1024
    
    # Encryption settings
    pbkdf2_iterations: int = 100000
    key_ttl: int = 30
    
    # RAG settings
    max_results: int = 10
    chunk_size: int = 4096
    
    # LLM settings
    model_name: str = "Qwen/Qwen3-0.6B"
    use_4bit: bool = True
    use_mock_llm: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            storage_path=data.get("storage", {}).get("path", "./data"),
            num_lanes=data.get("asic_simulator", {}).get("num_lanes", 256),
            batch_size=data.get("asic_simulator", {}).get("batch_size", 1024),
            pbkdf2_iterations=data.get("encryption", {}).get("pbkdf2_iterations", 100000),
            key_ttl=data.get("key_generator", {}).get("default_ttl", 30),
            max_results=data.get("rag_system", {}).get("max_results", 10),
            chunk_size=data.get("document_processing", {}).get("chunk_size", 4096),
            model_name=data.get("llm", {}).get("model_name", "Qwen/Qwen3-0.6B"),
            use_4bit=data.get("llm", {}).get("use_4bit", True),
        )


class ASICRAGSystem:
    """
    Main ASIC-RAG-CHIMERA system interface.
    
    Provides a unified API for document ingestion, querying,
    and system management.
    
    Example:
        >>> system = ASICRAGSystem(
        ...     storage_path="./data",
        ...     master_key=os.urandom(32)
        ... )
        >>> system.ingest_text("Q3 revenue was $45 million.", source="report.txt")
        >>> result = system.query("What was the Q3 revenue?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        storage_path: str = "./data",
        master_key: Optional[bytes] = None,
        config: Optional[SystemConfig] = None
    ):
        """
        Initialize ASIC-RAG-CHIMERA system.
        
        Args:
            storage_path: Path for persistent storage
            master_key: 256-bit encryption key (generated if not provided)
            config: System configuration
        """
        self.config = config or SystemConfig(storage_path=storage_path)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate master key if not provided
        self.master_key = master_key or os.urandom(32)
        
        # Initialize components
        self._init_components()
        
        # Statistics
        self._docs_ingested = 0
        self._queries_processed = 0
    
    def _init_components(self):
        """Initialize all system components."""
        from asic_simulator import GPUHashEngine, IndexManager, MerkleTree, KeyGenerator
        from rag_system import DocumentProcessor
        from rag_system.block_storage import BlockStorage, StorageConfig, BlockStorageManager
        from rag_system.query_engine import QueryEngine
        
        # ASIC simulator
        self.sha256_engine = GPUHashEngine() # Updated class name
        self.index_manager = IndexManager()
        self.merkle_tree = MerkleTree(self.sha256_engine) # Updated constructor
        self.key_generator = KeyGenerator(
            master_key=self.master_key,
            default_key_ttl=self.config.key_ttl
        )
        
        # Storage
        storage_config = StorageConfig(storage_path=str(self.storage_path / "blocks"))
        self.block_storage = BlockStorage(storage_config)
        self.storage_manager = BlockStorageManager(self.block_storage, self.master_key)
        
        # Document processor
        from rag_system.document_processor import ProcessingConfig
        proc_config = ProcessingConfig(chunk_size=self.config.chunk_size)
        self.document_processor = DocumentProcessor(proc_config)
        
        # LLM
        if self.config.use_mock_llm:
            from llm_interface import MockQwenLoader
            self.llm = MockQwenLoader()
        else:
            try:
                from llm_interface import QwenLoader, ModelConfig
                model_config = ModelConfig(
                    model_name=self.config.model_name,
                    use_4bit=self.config.use_4bit
                )
                self.llm = QwenLoader(model_config)
            except ImportError:
                from llm_interface import MockQwenLoader
                self.llm = MockQwenLoader()
        
        # Query engine with LLM generator
        def llm_generator(query: str, context: str) -> str:
            from llm_interface import ResponseGenerator
            generator = ResponseGenerator(self.llm)
            result = generator.generate(query, context)
            return result.answer
        
        self.query_engine = QueryEngine(
            self.storage_manager,
            self.index_manager,
            llm_generator=llm_generator
        )
    
    def ingest_text(
        self,
        content: str,
        source: str = "direct_input",
        tags: Optional[List[str]] = None,
        category: str = "general"
    ) -> List[int]:
        """
        Ingest text content into the system.
        
        Args:
            content: Document text content
            source: Source identifier
            tags: Additional tags for indexing
            category: Document category
            
        Returns:
            List of created block IDs
        """
        from rag_system.knowledge_block import BlockCategory
        
        # Map category string to enum
        category_map = {
            "general": BlockCategory.GENERAL,
            "finance": BlockCategory.FINANCE,
            "legal": BlockCategory.LEGAL,
            "technical": BlockCategory.TECHNICAL,
            "hr": BlockCategory.HR,
            "marketing": BlockCategory.MARKETING,
            "research": BlockCategory.RESEARCH,
        }
        block_category = category_map.get(category.lower(), BlockCategory.GENERAL)
        
        # Process document
        processed = self.document_processor.process_text(content, source, block_category)
        
        # Add custom tags
        if tags:
            processed.keywords.extend(tags)
        
        # Create blocks
        blocks = self.document_processor.create_blocks(processed)
        
        # Store and index
        block_ids = []
        for block in blocks:
            # Encrypt
            block.encrypt(self.master_key)
            
            # Store
            block_data = block.to_bytes()
            block_id = self.block_storage.write_block(block_data, tags=block.tag_hashes)
            block_ids.append(block_id)
            
            # Update index
            for tag_hash in block.tag_hashes:
                self.index_manager.add_tag(tag_hash, block_id, category)
            
            # Add to Merkle tree
            self.merkle_tree.add_leaf(block.header.block_hash)
        
        # Rebuild Merkle tree
        self.merkle_tree.build()
        
        self._docs_ingested += 1
        
        return block_ids
    
    def ingest_file(
        self,
        file_path: str,
        tags: Optional[List[str]] = None
    ) -> List[int]:
        """
        Ingest a document file.
        
        Args:
            file_path: Path to document file
            tags: Additional tags
            
        Returns:
            List of created block IDs
        """
        processed = self.document_processor.process_file(file_path)
        
        if tags:
            processed.keywords.extend(tags)
        
        blocks = self.document_processor.create_blocks(processed)
        
        block_ids = []
        for block in blocks:
            block.encrypt(self.master_key)
            block_data = block.to_bytes()
            block_id = self.block_storage.write_block(block_data, tags=block.tag_hashes)
            block_ids.append(block_id)
            
            for tag_hash in block.tag_hashes:
                self.index_manager.add_tag(tag_hash, block_id)
            
            self.merkle_tree.add_leaf(block.header.block_hash)
        
        self.merkle_tree.build()
        self._docs_ingested += 1
        
        return block_ids
    
    def query(
        self,
        question: str,
        category: Optional[str] = None,
        max_results: Optional[int] = None
    ):
        """
        Query the system.
        
        Args:
            question: Natural language question
            category: Filter by category
            max_results: Maximum results to return
            
        Returns:
            QueryResult with answer and sources
        """
        self._queries_processed += 1
        
        return self.query_engine.query(
            question,
            category_filter=category,
            generate_answer=True
        )
    
    def search(
        self,
        keywords: List[str],
        operation: str = "OR",
        limit: int = 10
    ) -> List[int]:
        """
        Search for blocks by keywords.
        
        Args:
            keywords: Search keywords
            operation: "AND" or "OR"
            limit: Maximum results
            
        Returns:
            List of matching block IDs
        """
        import hashlib
        from asic_simulator import SearchOperation
        
        tag_hashes = [
            hashlib.sha256(kw.encode()).digest()
            for kw in keywords
        ]
        
        op = SearchOperation.AND if operation.upper() == "AND" else SearchOperation.OR
        
        result = self.index_manager.search(tag_hashes, op, limit=limit)
        return result.block_ids
    
    def get_block(self, block_id: int) -> Optional[Dict]:
        """
        Retrieve and decrypt a block.
        
        Args:
            block_id: Block ID to retrieve
            
        Returns:
            Dictionary with block content and metadata
        """
        from rag_system import KnowledgeBlock
        
        block_data = self.block_storage.read_block(block_id)
        if not block_data:
            return None
        
        block = KnowledgeBlock.from_bytes(block_data)
        metadata, content = block.decrypt(self.master_key)
        
        return {
            "block_id": block_id,
            "content": content.decode('utf-8', errors='ignore'),
            "metadata": {
                "category": metadata.category.value,
                "source": metadata.source,
                "created_at": metadata.created_at,
            }
        }
    
    def verify_block(self, block_id: int) -> bool:
        """
        Verify block integrity using Merkle proof.
        
        Args:
            block_id: Block to verify
            
        Returns:
            True if block is valid
        """
        if block_id >= len(self.merkle_tree.leaves):
            return False
        
        proof = self.merkle_tree.get_proof(block_id)
        return self.merkle_tree.verify_proof(proof)
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            "documents_ingested": self._docs_ingested,
            "queries_processed": self._queries_processed,
            "total_blocks": self.block_storage.get_stats().total_blocks,
            "index_size": self.index_manager.get_statistics(),
            "merkle_leaves": len(self.merkle_tree.leaves),
            "storage_stats": {
                "read_ops": self.block_storage.get_stats().read_ops,
                "write_ops": self.block_storage.get_stats().write_ops,
            }
        }
    
    def close(self):
        """Close system and release resources."""
        self.block_storage.close()
        if hasattr(self.llm, 'unload'):
            self.llm.unload()


def create_system(
    storage_path: str = "./data",
    config_path: Optional[str] = None,
    master_key: Optional[bytes] = None
) -> ASICRAGSystem:
    """
    Factory function to create ASIC-RAG-CHIMERA system.
    
    Args:
        storage_path: Path for data storage
        config_path: Optional YAML config file
        master_key: Optional master encryption key
        
    Returns:
        Initialized ASICRAGSystem
    """
    config = None
    if config_path:
        config = SystemConfig.from_yaml(config_path)
        config.storage_path = storage_path
    
    return ASICRAGSystem(
        storage_path=storage_path,
        master_key=master_key,
        config=config
    )


if __name__ == "__main__":
    print("ASIC-RAG-CHIMERA System Demo")
    print("=" * 50)
    
    # Create system with mock LLM
    config = SystemConfig(use_mock_llm=True)
    system = ASICRAGSystem(
        storage_path="./demo_data",
        config=config
    )
    
    # Ingest documents
    print("\nIngesting documents...")
    
    system.ingest_text(
        "Q3 2024 Financial Report: Revenue was $45.2 million. "
        "Net profit margin improved to 23%. Employee count: 450.",
        source="q3_report.txt",
        category="finance"
    )
    
    system.ingest_text(
        "Technical Specification: API version 2.0 supports REST and GraphQL. "
        "Maximum throughput is 10,000 requests per second.",
        source="api_spec.txt",
        category="technical"
    )
    
    print(f"  Ingested {system._docs_ingested} documents")
    
    # Query system
    print("\nQuerying system...")
    
    result = system.query("What was the Q3 revenue?")
    
    print(f"\n  Query: What was the Q3 revenue?")
    print(f"  Answer: {result.answer}")
    print(f"  Results: {len(result.retrieved_blocks)}")
    print(f"  Time: {result.total_time_ms:.2f} ms")
    
    # Statistics
    print("\nSystem Statistics:")
    stats = system.get_statistics()
    print(f"  Documents: {stats['documents_ingested']}")
    print(f"  Queries: {stats['queries_processed']}")
    print(f"  Blocks: {stats['total_blocks']}")
    
    # Cleanup
    system.close()
    print("\nDemo complete!")
