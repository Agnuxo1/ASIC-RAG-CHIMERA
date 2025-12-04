"""
Temporary Key Generator for ASIC-RAG

Manages session-based encryption keys with automatic expiration.
Simulates ASIC hardware key management for secure RAG operations.

Security Features:
- Time-limited keys (default 30 seconds)
- Session isolation
- Cryptographic key derivation
- Automatic cleanup
"""

import hashlib
import hmac
import os
import secrets
import struct
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum


class KeyStatus(Enum):
    """Status of a temporary key."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class TemporaryKey:
    """
    Temporary decryption key with expiration.
    
    Attributes:
        key_id: Unique identifier for this key
        key_bytes: The actual key material (32 bytes for AES-256)
        block_id: ID of block this key can decrypt
        session_id: Session that owns this key
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        status: Current key status
        access_count: Number of times key was used
    """
    key_id: bytes
    key_bytes: bytes
    block_id: int
    session_id: bytes
    created_at: float
    expires_at: float
    status: KeyStatus = KeyStatus.ACTIVE
    access_count: int = 0
    
    @property
    def key_id_hex(self) -> str:
        return self.key_id.hex()
    
    @property
    def key_bytes_hex(self) -> str:
        return self.key_bytes.hex()
    
    @property
    def is_valid(self) -> bool:
        """Check if key is still valid."""
        return (
            self.status == KeyStatus.ACTIVE and
            time.time() < self.expires_at
        )
    
    @property
    def time_remaining(self) -> float:
        """Seconds until expiration."""
        return max(0, self.expires_at - time.time())
    
    def use(self) -> bool:
        """
        Attempt to use the key.
        
        Returns:
            True if key is valid and can be used
        """
        if not self.is_valid:
            return False
        self.access_count += 1
        return True
    
    def revoke(self):
        """Revoke this key immediately."""
        self.status = KeyStatus.REVOKED
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary (excluding actual key bytes)."""
        return {
            "key_id": self.key_id_hex,
            "block_id": self.block_id,
            "session_id": self.session_id.hex(),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "access_count": self.access_count,
            "time_remaining": self.time_remaining
        }


@dataclass
class KeySession:
    """
    Session for temporary key management.
    
    Groups related temporary keys under a single session
    for easier management and cleanup.
    """
    session_id: bytes
    created_at: float
    expires_at: float
    keys: Dict[bytes, TemporaryKey] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def session_id_hex(self) -> str:
        return self.session_id.hex()
    
    @property
    def is_active(self) -> bool:
        return time.time() < self.expires_at
    
    @property
    def key_count(self) -> int:
        return len(self.keys)
    
    @property
    def active_key_count(self) -> int:
        return sum(1 for k in self.keys.values() if k.is_valid)
    
    def add_key(self, key: TemporaryKey):
        """Add a key to this session."""
        self.keys[key.key_id] = key
    
    def get_key(self, key_id: bytes) -> Optional[TemporaryKey]:
        """Get a key by ID."""
        return self.keys.get(key_id)
    
    def revoke_all(self):
        """Revoke all keys in this session."""
        for key in self.keys.values():
            key.revoke()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired keys.
        
        Returns:
            Number of keys removed
        """
        expired_ids = [
            kid for kid, key in self.keys.items()
            if not key.is_valid
        ]
        for kid in expired_ids:
            del self.keys[kid]
        return len(expired_ids)


class KeyGenerator:
    """
    Temporary key generator for ASIC-RAG.
    
    Manages creation and lifecycle of session-based
    decryption keys with automatic expiration.
    
    Example:
        >>> generator = KeyGenerator(master_key=b"my_master_key")
        >>> session = generator.create_session()
        >>> temp_key = generator.generate_key(
        ...     session_id=session.session_id,
        ...     block_id=42,
        ...     block_hash=b"block_hash_here"
        ... )
        >>> if temp_key.is_valid:
        ...     # Use key for decryption
        ...     pass
    """
    
    def __init__(
        self,
        master_key: bytes,
        default_key_ttl: float = 30.0,
        default_session_ttl: float = 3600.0,
        cleanup_interval: float = 60.0,
        enable_auto_cleanup: bool = True
    ):
        """
        Initialize key generator.
        
        Args:
            master_key: Master encryption key
            default_key_ttl: Default key lifetime in seconds
            default_session_ttl: Default session lifetime in seconds
            cleanup_interval: Interval between cleanup runs
            enable_auto_cleanup: Enable automatic cleanup thread
        """
        self._master_key = master_key
        self._master_key_hash = hashlib.sha256(master_key).digest()
        self.default_key_ttl = default_key_ttl
        self.default_session_ttl = default_session_ttl
        
        # Session storage
        self._sessions: Dict[bytes, KeySession] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._total_keys_generated = 0
        self._total_sessions_created = 0
        self._total_keys_expired = 0
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        if enable_auto_cleanup:
            self._start_cleanup_thread(cleanup_interval)
    
    def _start_cleanup_thread(self, interval: float):
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.wait(interval):
                self._cleanup_expired()
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="KeyGenerator-Cleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_expired(self) -> Tuple[int, int]:
        """
        Clean up expired sessions and keys.
        
        Returns:
            Tuple of (sessions_removed, keys_removed)
        """
        with self._lock:
            sessions_removed = 0
            keys_removed = 0
            
            # Remove expired sessions
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if not session.is_active
            ]
            for sid in expired_sessions:
                keys_removed += len(self._sessions[sid].keys)
                del self._sessions[sid]
                sessions_removed += 1
            
            # Cleanup expired keys in active sessions
            for session in self._sessions.values():
                keys_removed += session.cleanup_expired()
            
            self._total_keys_expired += keys_removed
            return sessions_removed, keys_removed
    
    def create_session(
        self,
        ttl: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> KeySession:
        """
        Create a new key session.
        
        Args:
            ttl: Session lifetime in seconds (None = default)
            metadata: Optional session metadata
            
        Returns:
            New KeySession
        """
        with self._lock:
            session_id = secrets.token_bytes(32)
            now = time.time()
            ttl = ttl or self.default_session_ttl
            
            session = KeySession(
                session_id=session_id,
                created_at=now,
                expires_at=now + ttl,
                metadata=metadata or {}
            )
            
            self._sessions[session_id] = session
            self._total_sessions_created += 1
            
            return session
    
    def get_session(self, session_id: bytes) -> Optional[KeySession]:
        """Get session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                return session
            return None
    
    def generate_key(
        self,
        session_id: bytes,
        block_id: int,
        block_hash: bytes,
        ttl: Optional[float] = None
    ) -> Optional[TemporaryKey]:
        """
        Generate a temporary decryption key.
        
        Args:
            session_id: Session to associate key with
            block_id: ID of block this key will decrypt
            block_hash: Hash of the block for key derivation
            ttl: Key lifetime in seconds (None = default)
            
        Returns:
            TemporaryKey or None if session invalid
        """
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return None
            
            now = time.time()
            ttl = ttl or self.default_key_ttl
            
            # Generate unique key ID
            key_id = secrets.token_bytes(16)
            
            # Derive key from master key + block hash + session + nonce
            nonce = secrets.token_bytes(16)
            key_material = hmac.new(
                self._master_key,
                block_hash + session_id + nonce + struct.pack('>Q', block_id),
                hashlib.sha256
            ).digest()
            
            temp_key = TemporaryKey(
                key_id=key_id,
                key_bytes=key_material,
                block_id=block_id,
                session_id=session_id,
                created_at=now,
                expires_at=now + ttl
            )
            
            session.add_key(temp_key)
            self._total_keys_generated += 1
            
            return temp_key
    
    def generate_keys_batch(
        self,
        session_id: bytes,
        block_info: List[Tuple[int, bytes]],
        ttl: Optional[float] = None
    ) -> List[Optional[TemporaryKey]]:
        """
        Generate multiple temporary keys at once.
        
        Args:
            session_id: Session to associate keys with
            block_info: List of (block_id, block_hash) tuples
            ttl: Key lifetime in seconds
            
        Returns:
            List of TemporaryKey objects (None for failures)
        """
        return [
            self.generate_key(session_id, block_id, block_hash, ttl)
            for block_id, block_hash in block_info
        ]
    
    def get_key(
        self,
        session_id: bytes,
        key_id: bytes
    ) -> Optional[TemporaryKey]:
        """Get a specific key from a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return None
            
            key = session.get_key(key_id)
            if key and key.is_valid:
                return key
            return None
    
    def use_key(
        self,
        session_id: bytes,
        key_id: bytes
    ) -> Optional[bytes]:
        """
        Use a temporary key and return its bytes.
        
        Args:
            session_id: Session owning the key
            key_id: Key identifier
            
        Returns:
            Key bytes if valid, None otherwise
        """
        with self._lock:
            key = self.get_key(session_id, key_id)
            if key and key.use():
                return key.key_bytes
            return None
    
    def revoke_key(self, session_id: bytes, key_id: bytes):
        """Revoke a specific key."""
        with self._lock:
            session = self.get_session(session_id)
            if session:
                key = session.get_key(key_id)
                if key:
                    key.revoke()
    
    def revoke_session(self, session_id: bytes):
        """Revoke all keys in a session."""
        with self._lock:
            session = self.get_session(session_id)
            if session:
                session.revoke_all()
    
    def close_session(self, session_id: bytes):
        """Close and remove a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].revoke_all()
                del self._sessions[session_id]
    
    def get_statistics(self) -> Dict:
        """Get generator statistics."""
        with self._lock:
            active_sessions = sum(
                1 for s in self._sessions.values()
                if s.is_active
            )
            active_keys = sum(
                s.active_key_count for s in self._sessions.values()
            )
            
            return {
                "total_keys_generated": self._total_keys_generated,
                "total_sessions_created": self._total_sessions_created,
                "total_keys_expired": self._total_keys_expired,
                "active_sessions": active_sessions,
                "active_keys": active_keys,
                "default_key_ttl": self.default_key_ttl,
                "default_session_ttl": self.default_session_ttl
            }
    
    def shutdown(self):
        """Shutdown generator and cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        with self._lock:
            for session in self._sessions.values():
                session.revoke_all()
            self._sessions.clear()


class SecureKeyDerivation:
    """
    Secure key derivation utilities.
    
    Provides PBKDF2-based key derivation for master keys
    and HKDF-style expansion for temporary keys.
    """
    
    @staticmethod
    def derive_master_key(
        password: str,
        salt: bytes,
        iterations: int = 100000
    ) -> bytes:
        """
        Derive master key from password using PBKDF2.
        
        Args:
            password: User password
            salt: Random salt (should be stored)
            iterations: PBKDF2 iterations
            
        Returns:
            32-byte derived key
        """
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations,
            dklen=32
        )
    
    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Generate random salt."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def expand_key(
        prk: bytes,
        info: bytes,
        length: int = 32
    ) -> bytes:
        """
        Expand pseudo-random key using HKDF-Expand.
        
        Args:
            prk: Pseudo-random key
            info: Context information
            length: Output length in bytes
            
        Returns:
            Expanded key material
        """
        # HKDF-Expand
        hash_len = 32  # SHA-256
        n = (length + hash_len - 1) // hash_len
        
        okm = b""
        t = b""
        
        for i in range(1, n + 1):
            t = hmac.new(
                prk,
                t + info + bytes([i]),
                hashlib.sha256
            ).digest()
            okm += t
        
        return okm[:length]


if __name__ == "__main__":
    print("Key Generator Demo")
    print("=" * 50)
    
    # Create generator with master key
    master_key = SecureKeyDerivation.derive_master_key(
        password="demo_password",
        salt=SecureKeyDerivation.generate_salt()
    )
    
    generator = KeyGenerator(
        master_key=master_key,
        default_key_ttl=5.0,  # 5 seconds for demo
        enable_auto_cleanup=False
    )
    
    # Create session
    session = generator.create_session()
    print(f"\nSession created: {session.session_id_hex[:16]}...")
    
    # Generate some keys
    print("\nGenerating temporary keys:")
    block_hashes = [hashlib.sha256(f"block_{i}".encode()).digest() for i in range(3)]
    
    keys = []
    for i, block_hash in enumerate(block_hashes):
        key = generator.generate_key(
            session_id=session.session_id,
            block_id=i,
            block_hash=block_hash
        )
        if key:
            keys.append(key)
            print(f"  Block {i}: Key {key.key_id_hex[:8]}... (TTL: {key.time_remaining:.1f}s)")
    
    # Use a key
    print("\nUsing key 0:")
    key_bytes = generator.use_key(session.session_id, keys[0].key_id)
    if key_bytes:
        print(f"  Key bytes: {key_bytes.hex()[:32]}...")
        print(f"  Access count: {keys[0].access_count}")
    
    # Wait for expiration
    print("\nWaiting 6 seconds for keys to expire...")
    time.sleep(6)
    
    # Try to use expired key
    print("\nTrying to use expired key:")
    key_bytes = generator.use_key(session.session_id, keys[0].key_id)
    if key_bytes:
        print("  Key still valid!")
    else:
        print("  Key expired as expected")
    
    # Statistics
    print("\n--- Statistics ---")
    stats = generator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    generator.shutdown()
