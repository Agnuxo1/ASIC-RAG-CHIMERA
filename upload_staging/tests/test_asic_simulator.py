"""
Unit Tests for ASIC Simulator Module

Tests:
- SHA-256 engine correctness
- Merkle tree construction and verification
- Index manager operations
- Key generation and expiration

Run:
    pytest tests/test_asic_simulator.py -v
"""

import pytest
import hashlib
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSHA256Engine:
    """Tests for SHA-256 engine."""
    
    def test_single_hash_correctness(self):
        """Test that single hash matches hashlib."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        test_data = b"Hello, World!"
        expected = hashlib.sha256(test_data).digest()
        
        result = engine.hash(test_data)
        
        assert result.hash_bytes == expected
    
    def test_batch_hash_correctness(self):
        """Test batch hash correctness."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        test_data = [f"data_{i}".encode() for i in range(100)]
        expected = [hashlib.sha256(d).digest() for d in test_data]
        
        result = engine.hash_batch(test_data)
        
        for i, r in enumerate(result.results):
            assert r.hash_bytes == expected[i]
    
    def test_double_hash(self):
        """Test Bitcoin-style double hash."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        data = b"test"
        expected = hashlib.sha256(hashlib.sha256(data).digest()).digest()
        
        result = engine.double_hash(data)
        
        assert result == expected
    
    def test_hash_concat(self):
        """Test hash concatenation for Merkle trees."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        left = hashlib.sha256(b"left").digest()
        right = hashlib.sha256(b"right").digest()
        expected = hashlib.sha256(left + right).digest()
        
        result = engine.hash_concat(left, right)
        
        assert result == expected
    
    def test_verify_hash(self):
        """Test hash verification."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        data = b"verify me"
        hash_bytes = hashlib.sha256(data).digest()
        
        assert engine.verify_hash(data, hash_bytes) is True
        assert engine.verify_hash(data, b'\x00' * 32) is False
    
    def test_parallel_lanes(self):
        """Test parallel lane configuration."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine(num_lanes=128)
        
        assert engine.num_lanes == 128
    
    def test_cache_functionality(self):
        """Test hash caching."""
        from asic_simulator import SHA256Engine
        
        engine = SHA256Engine()
        
        data = b"cached data"
        
        # First hash
        result1 = engine.hash(data)
        
        # Second hash should hit cache
        result2 = engine.hash(data)
        
        assert result1.hash_bytes == result2.hash_bytes
        
        metrics = engine.get_metrics()
        assert metrics.cache_hits >= 1


class TestMerkleTree:
    """Tests for Merkle tree."""
    
    def test_single_leaf(self):
        """Test tree with single leaf."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        tree.add_leaf(b"single leaf")
        tree.build()
        
        assert tree.root is not None
        assert tree.leaf_count == 1
    
    def test_multiple_leaves(self):
        """Test tree with multiple leaves."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        for i in range(8):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.build()
        
        assert tree.root is not None
        assert tree.leaf_count == 8
    
    def test_odd_leaves(self):
        """Test tree with odd number of leaves."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        for i in range(7):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.build()
        
        assert tree.root is not None
        assert tree.leaf_count == 7
    
    def test_proof_generation(self):
        """Test Merkle proof generation."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        for i in range(16):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.build()
        
        # Get proof for leaf 5
        proof = tree.get_proof(5)
        
        assert proof is not None
        assert len(proof.proof_path) > 0
    
    def test_proof_verification(self):
        """Test Merkle proof verification."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        for i in range(16):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.build()
        
        # Get and verify proof
        proof = tree.get_proof(5)
        
        assert tree.verify_proof(proof) is True
    
    def test_invalid_proof(self):
        """Test invalid proof detection."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        for i in range(16):
            tree.add_leaf(f"leaf_{i}".encode())
        tree.build()
        
        proof = tree.get_proof(5)
        
        # Corrupt the proof
        proof.leaf_hash = b'\x00' * 32
        
        assert tree.verify_proof(proof) is False
    
    def test_find_leaf_by_hash(self):
        """Test finding leaf by hash."""
        from asic_simulator import MerkleTree
        
        tree = MerkleTree()
        target_data = b"find me"
        
        tree.add_leaf(b"other1")
        tree.add_leaf(target_data)
        tree.add_leaf(b"other2")
        tree.build()
        
        target_hash = hashlib.sha256(target_data).digest()
        node = tree.find_leaf_by_hash(target_hash)
        
        assert node is not None
        assert node.data == target_data


class TestIndexManager:
    """Tests for index manager."""
    
    def test_add_tag(self):
        """Test adding a tag."""
        from asic_simulator import IndexManager
        
        manager = IndexManager()
        
        tag = b'\x01' * 32
        block_id = 42
        
        manager.add_tag(tag, block_id)
        
        blocks = manager.get_blocks_for_tag(tag)
        assert block_id in blocks
    
    def test_add_multiple_tags(self):
        """Test adding multiple tags."""
        from asic_simulator import IndexManager
        
        manager = IndexManager()
        
        tags = [bytes([i] * 32) for i in range(5)]
        block_id = 1
        
        manager.add_tags(tags, block_id)
        
        for tag in tags:
            assert block_id in manager.get_blocks_for_tag(tag)
    
    def test_search_and(self):
        """Test AND search."""
        from asic_simulator import IndexManager, SearchOperation
        
        manager = IndexManager()
        
        tag1 = b'\x01' * 32
        tag2 = b'\x02' * 32
        
        # Block 1 has both tags
        manager.add_tag(tag1, 1)
        manager.add_tag(tag2, 1)
        
        # Block 2 has only tag1
        manager.add_tag(tag1, 2)
        
        result = manager.search([tag1, tag2], SearchOperation.AND)
        
        assert 1 in result.block_ids
        assert 2 not in result.block_ids
    
    def test_search_or(self):
        """Test OR search."""
        from asic_simulator import IndexManager, SearchOperation
        
        manager = IndexManager()
        
        tag1 = b'\x01' * 32
        tag2 = b'\x02' * 32
        
        manager.add_tag(tag1, 1)
        manager.add_tag(tag2, 2)
        
        result = manager.search([tag1, tag2], SearchOperation.OR)
        
        assert 1 in result.block_ids
        assert 2 in result.block_ids
    
    def test_remove_tag(self):
        """Test removing a tag."""
        from asic_simulator import IndexManager
        
        manager = IndexManager()
        
        tag = b'\x01' * 32
        manager.add_tag(tag, 1)
        manager.remove_tag(tag, 1)
        
        blocks = manager.get_blocks_for_tag(tag)
        assert 1 not in blocks
    
    def test_get_tags_for_block(self):
        """Test reverse lookup."""
        from asic_simulator import IndexManager
        
        manager = IndexManager()
        
        tags = [bytes([i] * 32) for i in range(3)]
        block_id = 1
        
        for tag in tags:
            manager.add_tag(tag, block_id)
        
        result = manager.get_tags_for_block(block_id)
        
        for tag in tags:
            assert tag in result


class TestKeyGenerator:
    """Tests for key generator."""
    
    def test_create_session(self):
        """Test session creation."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key)
        
        session = generator.create_session()
        
        assert session is not None
        assert session.session_id is not None
    
    def test_generate_key(self):
        """Test key generation."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key)
        
        session = generator.create_session()
        block_hash = os.urandom(32)
        block_id = 42
        
        key = generator.generate_key(session.session_id, block_id, block_hash)
        
        assert key is not None
        assert len(key.key_bytes) == 32
    
    def test_key_expiration(self):
        """Test key expiration."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key, default_key_ttl=0.1)
        
        session = generator.create_session()
        block_hash = os.urandom(32)
        block_id = 1
        
        key = generator.generate_key(session.session_id, block_id, block_hash)
        
        # Key should be valid initially
        assert key.is_valid is True
        
        # Wait for expiration
        time.sleep(0.15)
        
        assert key.is_valid is False
    
    def test_key_revocation(self):
        """Test key revocation."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key)
        
        session = generator.create_session()
        block_hash = os.urandom(32)
        block_id = 1
        
        key = generator.generate_key(session.session_id, block_id, block_hash)
        
        assert key.is_valid is True
        
        generator.revoke_key(session.session_id, key.key_id)
        
        assert key.is_valid is False
    
    def test_session_close(self):
        """Test session closing."""
        from asic_simulator import KeyGenerator
        
        master_key = os.urandom(32)
        generator = KeyGenerator(master_key=master_key)
        
        session = generator.create_session()
        
        generator.close_session(session.session_id)
        
        # Generating key for closed session should fail
        block_hash = os.urandom(32)
        block_id = 1
        key = generator.generate_key(session.session_id, block_id, block_hash)
        
        assert key is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
