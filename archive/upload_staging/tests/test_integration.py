"""
Integration Tests for ASIC-RAG-CHIMERA

End-to-end tests verifying:
- Complete document ingestion pipeline
- Full query processing
- Component integration
- Security properties

Run:
    pytest tests/test_integration.py -v
"""

import pytest
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFullPipeline:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def system_setup(self):
        """Set up complete system for testing."""
        from rag_system.block_storage import BlockStorage, StorageConfig, BlockStorageManager
        from asic_simulator import IndexManager, MerkleTree, GPUHashEngine
        from rag_system import DocumentProcessor
        
        tmp_dir = tempfile.mkdtemp()
        master_key = os.urandom(32)
        
        config = StorageConfig(storage_path=tmp_dir, max_db_size=10*1024*1024) # 10MB for tests
        storage = BlockStorage(config)
        storage_manager = BlockStorageManager(storage, master_key)
        index = IndexManager()
        hash_engine = GPUHashEngine()
        merkle = MerkleTree(hash_engine)
        processor = DocumentProcessor()
        processor = DocumentProcessor()
        
        yield {
            "storage": storage,
            "storage_manager": storage_manager,
            "index": index,
            "merkle": merkle,
            "processor": processor,
            "master_key": master_key,
            "tmp_dir": tmp_dir
        }
        
        storage.close()
    
    def test_ingest_and_retrieve(self, system_setup):
        """Test complete document ingestion and retrieval."""
        storage = system_setup["storage"]
        processor = system_setup["processor"]
        master_key = system_setup["master_key"]
        index = system_setup["index"]
        
        # Create test document
        doc_content = """
        Q3 2024 Financial Report
        
        Revenue reached $45.2 million this quarter.
        Net profit margin improved to 23%.
        The company hired 50 new employees.
        """
        
        # Process document
        processed = processor.process_text(doc_content, "q3_report.txt")
        
        # Create and store blocks
        blocks = processor.create_blocks(processed)
        
        block_ids = []
        for block in blocks:
            block.encrypt(master_key)
            block_data = block.to_bytes()
            
            block_id = storage.write_block(block_data, tags=block.tag_hashes)
            block_ids.append(block_id)
            
            # Update index
            for tag_hash in block.tag_hashes:
                index.add_tag(tag_hash, block_id)
        
        # Verify storage
        assert len(block_ids) >= 1
        
        # Retrieve and decrypt
        for block_id in block_ids:
            block_data = storage.read_block(block_id)
            assert block_data is not None
    
    def test_search_pipeline(self, system_setup):
        """Test search through indexed documents."""
        storage = system_setup["storage"]
        processor = system_setup["processor"]
        master_key = system_setup["master_key"]
        index = system_setup["index"]
        
        # Ingest multiple documents
        documents = [
            ("Finance Report: Revenue was $100 million.", "finance"),
            ("Technical Spec: API supports REST and GraphQL.", "technical"),
            ("HR Policy: Vacation days increased to 25.", "hr"),
        ]
        
        from asic_simulator import SearchOperation
        import hashlib
        
        for content, category in documents:
            processed = processor.process_text(content, f"{category}.txt")
            blocks = processor.create_blocks(processed)
            
            for block in blocks:
                block.encrypt(master_key)
                block_id = storage.write_block(block.to_bytes(), tags=block.tag_hashes)
                
                for tag_hash in block.tag_hashes:
                    index.add_tag(tag_hash, block_id)
        
        # Search for finance documents
        finance_tag = hashlib.sha256(b"revenue").digest()
        result = index.search([finance_tag], SearchOperation.OR)
        
        assert len(result.block_ids) >= 1
    
    def test_merkle_integration(self, system_setup):
        """Test Merkle tree integration with storage."""
        storage = system_setup["storage"]
        merkle = system_setup["merkle"]
        
        # Store blocks and add to Merkle tree
        block_ids = []
        for i in range(10):
            block_data = f"Block content {i}".encode()
            block_id = storage.write_block(block_data)
            block_ids.append(block_id)
            
            import hashlib
            block_hash = hashlib.sha256(block_data).digest()
            merkle.add_leaf(block_hash)
        
        # Build tree
        merkle.build()
        
        # Verify proofs for all blocks
        for i, block_id in enumerate(block_ids):
            proof = merkle.get_proof(i)
            assert merkle.verify_proof(proof) is True
    
    def test_key_rotation(self, system_setup):
        """Test key rotation and re-encryption."""
        storage = system_setup["storage"]
        
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        old_key = os.urandom(32)
        new_key = os.urandom(32)
        
        # Create and encrypt block with old key
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="test.txt")
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=b"Sensitive data"
        )
        
        block.encrypt(old_key)
        block_data = block.to_bytes()
        block_id = storage.write_block(block_data)
        
        # Retrieve, decrypt with old key, re-encrypt with new key
        stored_data = storage.read_block(block_id)
        restored_block = KnowledgeBlock.from_bytes(stored_data)
        
        _, content = restored_block.decrypt(old_key)
        
        # Re-encrypt with new key
        restored_block._is_encrypted = False
        restored_block.content = content
        restored_block.encrypt(new_key)
        
        # Verify decryption with new key works
        _, decrypted = restored_block.decrypt(new_key)
        assert decrypted == b"Sensitive data"


class TestSecurityProperties:
    """Tests for security properties."""
    
    def test_encryption_confidentiality(self):
        """Test that encrypted data doesn't leak plaintext."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        plaintext = b"Super secret password: hunter2"
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="secrets.txt")
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=plaintext
        )
        
        master_key = os.urandom(32)
        block.encrypt(master_key)
        
        encrypted_data = block.to_bytes()
        
        # Plaintext should not appear in encrypted data
        assert plaintext not in encrypted_data
        assert b"hunter2" not in encrypted_data
    
    def test_wrong_key_fails(self):
        """Test that wrong key fails decryption."""
        from rag_system import KnowledgeBlock, BlockHeader, BlockMetadata, BlockCategory
        
        header = BlockHeader(prev_hash=b'\x00' * 32, timestamp=time.time(), nonce=0)
        metadata = BlockMetadata(category=BlockCategory.GENERAL, source="test.txt")
        
        block = KnowledgeBlock(
            block_id=1,
            header=header,
            tag_hashes=[],
            metadata=metadata,
            content=b"Secret content"
        )
        
        correct_key = os.urandom(32)
        wrong_key = os.urandom(32)
        
        block.encrypt(correct_key)
        
        with pytest.raises(Exception):
            block.decrypt(wrong_key)
    
    def test_tag_hash_opacity(self):
        """Test that tag hashes don't reveal keywords."""
        import hashlib
        
        keyword = "confidential_project_name"
        tag_hash = hashlib.sha256(keyword.encode()).digest()
        
        # Hash should not contain readable keyword
        assert keyword.encode() not in tag_hash
        
        # Hash should be fixed 32 bytes
        assert len(tag_hash) == 32
    
    def test_key_expiration_enforced(self):
        """Test that expired keys cannot be used."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key, default_key_ttl=0.1)
        
        session = generator.create_session()
        block_hash = os.urandom(32)
        block_id = 1
        
        key = generator.generate_key(session.session_id, block_id, block_hash)
        
        # Use key immediately
        key_bytes = generator.use_key(session.session_id, key.key_id)
        assert key_bytes is not None
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Try to use expired key
        key_bytes = generator.use_key(session.session_id, key.key_id)
        assert key_bytes is None


class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_hash_throughput_minimum(self):
        """Ensure hash throughput meets minimum requirement."""
        from asic_simulator import GPUHashEngine
        
        engine = GPUHashEngine()
        
        test_data = [f"data_{i}".encode() for i in range(1000)]
        
        result = engine.hash_batch(test_data)
        
        # Should achieve at least 100K hashes/second
        assert result.hashes_per_second >= 100000
    
    def test_search_latency_maximum(self):
        """Ensure search latency meets maximum requirement."""
        from asic_simulator import IndexManager, SearchOperation
        
        index = IndexManager()
        
        # Build index with 1000 blocks
        for block_id in range(1000):
            tags = [os.urandom(32) for _ in range(5)]
            for tag in tags:
                index.add_tag(tag, block_id)
        
        # Measure search latency
        import hashlib
        search_tags = [hashlib.sha256(f"tag_{i}".encode()).digest() for i in range(3)]
        
        start = time.perf_counter()
        for _ in range(100):
            index.search(search_tags, SearchOperation.OR)
        end = time.perf_counter()
        
        avg_latency_ms = (end - start) * 1000 / 100
        
        # Should complete in under 10ms
        assert avg_latency_ms < 10
    
    def test_encryption_throughput_minimum(self):
        """Ensure encryption throughput meets minimum requirement."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        key = os.urandom(32)
        aesgcm = AESGCM(key)
        
        data = os.urandom(4096)
        
        start = time.perf_counter()
        for _ in range(1000):
            nonce = os.urandom(12)
            aesgcm.encrypt(nonce, data, None)
        end = time.perf_counter()
        
        throughput_mbps = (4096 * 1000) / (1024 * 1024) / (end - start)
        
        # Should achieve at least 100 MB/s
        assert throughput_mbps >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
