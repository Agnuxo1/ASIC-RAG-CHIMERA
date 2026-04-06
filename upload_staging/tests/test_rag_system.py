"""
Unit Tests for RAG System Module

Tests:
- Knowledge block creation and encryption
- Block storage operations
- Document processor
- Query engine

Run:
    pytest tests/test_rag_system.py -v
"""

import pytest
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKnowledgeBlock:
    """Tests for knowledge block."""
    
    def test_block_creation(self):
        """Test block creation."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(
            prev_hash=b'\x00' * 32,
            timestamp=time.time(),
            nonce=0
        )
        
        metadata = BlockMetadata(
            category=BlockCategory.GENERAL,
            source="test.txt"
        )
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[os.urandom(32) for _ in range(3)],
            metadata=metadata,
            content=b"Test content"
        )
        
        assert block.block_id == 1
        assert block.content == b"Test content"
        assert len(block.tag_hashes) == 3
    
    def test_block_encryption(self):
        """Test block encryption."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="test.txt")
        
        original_content = b"Secret content that should be encrypted"
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=original_content
        )
        
        master_key = os.urandom(32)
        block.encrypt(master_key)
        
        assert block._is_encrypted is True
        assert block._encrypted_payload is not None
    
    def test_block_decryption(self):
        """Test block decryption."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.FINANCE, source="report.txt")
        
        original_content = b"Confidential financial data"
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=original_content
        )
        
        master_key = os.urandom(32)
        block.encrypt(master_key)
        
        decrypted_metadata, decrypted_content = block.decrypt(master_key)
        
        assert decrypted_content == original_content
        assert decrypted_metadata.category == BlockCategory.FINANCE
    
    def test_block_serialization(self):
        """Test block serialization."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="test.txt")
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[os.urandom(32)],
            metadata=metadata,
            content=b"Test content"
        )
        
        master_key = os.urandom(32)
        block.encrypt(master_key)
        
        # Serialize
        block_bytes = block.to_bytes()
        
        assert len(block_bytes) > 0
        
        # Deserialize (pass block_id since it's not stored in serialization)
        restored = KnowledgeBlock.from_bytes(block_bytes, block_id=1)
        
        assert restored.block_id == block.block_id
    
    def test_block_integrity(self):
        """Test block integrity verification."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="test.txt")
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=b"Test content"
        )
        
        # Before tampering
        assert block.verify_integrity() is True
        
        # After tampering
        block.content = b"Modified content"
        assert block.verify_integrity() is False


class TestBlockStorage:
    """Tests for block storage."""
    
    def test_write_and_read(self):
        """Test writing and reading blocks."""
        from rag_system.block_storage import BlockStorage, StorageConfig
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = StorageConfig(storage_path=tmp_dir, max_db_size=1024*1024)  # 1MB
            storage = BlockStorage(config)
            
            block_data = b"Test block data"
            tags = [os.urandom(32) for _ in range(2)]
            
            block_id = storage.write_block(block_data, tags=tags)
            
            assert block_id is not None
            
            read_data = storage.read_block(block_id)
            
            assert read_data == block_data
            
            storage.close()
    
    def test_search_by_tags(self):
        """Test searching blocks by tags."""
        from rag_system.block_storage import BlockStorage, StorageConfig
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = StorageConfig(storage_path=tmp_dir, max_db_size=1024*1024)  # 1MB
            storage = BlockStorage(config)
            
            common_tag = os.urandom(32)
            unique_tag = os.urandom(32)
            
            # Block 1 has common tag
            storage.write_block(b"Block 1", tags=[common_tag])
            
            # Block 2 has both tags
            storage.write_block(b"Block 2", tags=[common_tag, unique_tag])
            
            # Search by common tag
            results = storage.search_by_tags([common_tag])
            assert len(results) == 2
            
            # Search by unique tag
            results = storage.search_by_tags([unique_tag])
            assert len(results) == 1
            
            storage.close()
    
    def test_delete_block(self):
        """Test deleting blocks."""
        from rag_system.block_storage import BlockStorage, StorageConfig
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = StorageConfig(storage_path=tmp_dir, max_db_size=1024*1024)  # 1MB
            storage = BlockStorage(config)
            
            block_id = storage.write_block(b"To be deleted")
            
            assert storage.read_block(block_id) is not None
            
            storage.delete_block(block_id)
            
            assert storage.read_block(block_id) is None
            
            storage.close()
    
    def test_storage_stats(self):
        """Test storage statistics."""
        from rag_system.block_storage import BlockStorage, StorageConfig
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = StorageConfig(storage_path=tmp_dir, max_db_size=1024*1024)  # 1MB
            storage = BlockStorage(config)
            
            for i in range(10):
                storage.write_block(f"Block {i}".encode())
            
            stats = storage.get_stats()
            
            assert stats.total_blocks == 10
            assert stats.write_ops == 10
            
            storage.close()


class TestDocumentProcessor:
    """Tests for document processor."""
    
    def test_process_text(self):
        """Test processing text content."""
        from rag_system import DocumentProcessor
        from rag_system.knowledge_block import BlockCategory
        
        processor = DocumentProcessor()
        
        text = """
        Q3 2024 Financial Report
        
        Revenue increased by 15% to $45.2 million.
        Net profit margin improved to 23%.
        Employee count grew to 450.
        """
        
        result = processor.process_text(text, "report.txt", BlockCategory.FINANCE)
        
        assert result.content == text
        assert len(result.keywords) > 0
        assert "revenue" in result.keywords or "financial" in result.keywords
    
    def test_chunking(self):
        """Test document chunking."""
        from rag_system import DocumentProcessor
        from rag_system.document_processor import ProcessingConfig
        
        config = ProcessingConfig(chunk_size=100, chunk_overlap=20)
        processor = DocumentProcessor(config)
        
        # Create long text
        text = "This is a test sentence. " * 50
        
        result = processor.process_text(text, "long.txt")
        
        assert len(result.chunks) > 1
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        from rag_system.document_processor import KeywordExtractor
        
        extractor = KeywordExtractor(min_length=3, max_keywords=10)
        
        text = "Machine learning and artificial intelligence are transforming technology. AI systems use algorithms."
        
        keywords = extractor.extract(text)
        
        assert len(keywords) > 0
        assert "machine" in keywords or "learning" in keywords or "artificial" in keywords
    
    def test_create_blocks(self):
        """Test creating blocks from processed document."""
        from rag_system import DocumentProcessor
        
        processor = DocumentProcessor()
        
        text = "Test document content for block creation."
        result = processor.process_text(text, "test.txt")
        
        blocks = processor.create_blocks(result)
        
        assert len(blocks) >= 1
        assert blocks[0].content is not None


class TestQueryEngine:
    """Tests for query engine."""
    
    def test_query_keyword_extraction(self):
        """Test extracting keywords from queries."""
        from rag_system.query_engine import QueryEngine
        from rag_system.document_processor import KeywordExtractor
        
        # Create mock objects
        class MockStorage:
            def retrieve_block(self, block_id):
                return None
        
        class MockIndex:
            def search(self, tags, operation, limit, min_relevance, category):
                return type('Result', (), {
                    'block_ids': [],
                    'relevance_scores': {},
                    'total_candidates': 0
                })()
        
        engine = QueryEngine(MockStorage(), MockIndex())
        
        keywords = engine._extract_query_keywords("What was the Q3 revenue growth?")
        
        assert len(keywords) > 0
    
    def test_context_assembly(self):
        """Test assembling context from blocks."""
        from rag_system.query_engine import QueryEngine, RetrievedBlock
        from rag_system.knowledge_block import BlockMetadata, BlockCategory
        
        class MockStorage:
            def retrieve_block(self, block_id):
                return None
        
        class MockIndex:
            def search(self, *args, **kwargs):
                return type('Result', (), {'block_ids': [], 'relevance_scores': {}, 'total_candidates': 0})()
        
        engine = QueryEngine(MockStorage(), MockIndex())
        
        blocks = [
            RetrievedBlock(
                block_id=1,
                content="Revenue was $45 million.",
                metadata=BlockMetadata(category=BlockCategory.FINANCE, source="report.txt"),
                relevance_score=0.9,
                matched_tags=["revenue"],
                decryption_time_ms=1.0
            ),
            RetrievedBlock(
                block_id=2,
                content="Employee count is 450.",
                metadata=BlockMetadata(category=BlockCategory.HR, source="hr.txt"),
                relevance_score=0.7,
                matched_tags=["employee"],
                decryption_time_ms=1.0
            ),
        ]
        
        context = engine._assemble_context(blocks)
        
        assert "Revenue" in context
        assert "Employee" in context
        assert "[Document 1]" in context


class TestBlockChain:
    """Tests for blockchain structure."""
    
    def test_chain_creation(self):
        """Test creating a blockchain."""
        from rag_system.knowledge_block import BlockChain, KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        chain = BlockChain()
        
        for i in range(5):
            header = BlockHeader(
                prev_hash=chain.get_latest_hash() if chain.blocks else b'\x00' * 32,
                timestamp=time.time(),
                nonce=i
            )
            
            block = KnowledgeBlock(
                block_id=i,
                header=header,
                tag_hashes=[],
                metadata=BlockMetadata(category=BlockCategory.GENERAL, source=f"doc_{i}.txt"),
                content=f"Block {i} content".encode()
            )
            
            chain.add_block(block)
        
        assert len(chain.blocks) == 5
    
    def test_chain_verification(self):
        """Test chain integrity verification."""
        from rag_system.knowledge_block import BlockChain, KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        chain = BlockChain()
        
        prev_hash = b'\x00' * 32
        for i in range(3):
            header = BlockHeader(prev_hash=prev_hash, timestamp=time.time(), nonce=i)
            
            block = KnowledgeBlock(
                block_id=i,
                header=header,
                tag_hashes=[],
                metadata=BlockMetadata(category=BlockCategory.GENERAL, source=f"doc_{i}.txt"),
                content=f"Block {i}".encode()
            )
            
            chain.add_block(block)
            prev_hash = header.block_hash
        
        assert chain.verify_chain() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
