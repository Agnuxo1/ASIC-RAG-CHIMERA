# ASIC-RAG-CHIMERA Security Model

## Overview

ASIC-RAG-CHIMERA implements a comprehensive security model designed to protect sensitive documents while enabling efficient retrieval and AI-powered responses. This document details the security architecture, threat model, and cryptographic implementations.

## Security Objectives

1. **Confidentiality**: Document contents are encrypted at rest
2. **Integrity**: Cryptographic verification of document authenticity
3. **Privacy**: Search keywords are hashed before indexing
4. **Access Control**: Time-limited keys for document access
5. **Auditability**: Complete audit trail of all operations

## Cryptographic Primitives

### Encryption: AES-256-GCM

All document content is encrypted using AES-256 in Galois/Counter Mode (GCM).

**Properties:**
- 256-bit key size (128-bit security level)
- Authenticated encryption (integrity + confidentiality)
- 12-byte random nonce per encryption
- 16-byte authentication tag

**Implementation:**
```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)
nonce = os.urandom(12)
ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
```

### Key Derivation: PBKDF2

Block-specific keys are derived using PBKDF2-HMAC-SHA256.

**Parameters:**
- Iterations: 100,000 (adjustable)
- Salt: 32 bytes random per derivation
- Output: 32 bytes (256 bits)

**Implementation:**
```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000
)
derived_key = kdf.derive(password)
```

### Hashing: SHA-256

All cryptographic hashing uses SHA-256.

**Uses:**
- Tag/keyword hashing for opaque indexing
- Block hash for integrity verification
- Merkle tree construction
- Session key derivation

## Key Management

### Key Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                     Master Key                          │
│              (User-provided, 256 bits)                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Block Keys                            │
│    PBKDF2(master_key || block_hash || salt)            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Session Keys                           │
│   HMAC(master_key, session_id || block_hash || nonce)  │
│              (30-second TTL)                            │
└─────────────────────────────────────────────────────────┘
```

### Temporary Key Lifecycle

1. **Generation**: Key created for specific block access
2. **Active**: Key can be used for encryption/decryption
3. **Expiration**: Key automatically expires after TTL (30s default)
4. **Revocation**: Key can be manually revoked
5. **Cleanup**: Expired keys are removed by background thread

### Key Security Properties

| Property | Implementation |
|----------|----------------|
| Secrecy | Keys never stored in plaintext |
| Uniqueness | Each block has unique derived key |
| Forward Secrecy | Session keys expire, limiting exposure |
| Key Rotation | Supported via re-encryption |

## Privacy Protection

### Opaque Tag Indexing

Keywords are never stored in plaintext. Before indexing:

```python
# Keyword is hashed before any storage
tag_hash = hashlib.sha256(keyword.encode()).digest()
index.add_tag(tag_hash, block_id)
```

**Privacy Properties:**
- Original keywords cannot be recovered from index
- Search requires knowing the exact keyword
- Similar keywords produce completely different hashes

### Local LLM Processing

All AI processing happens locally:

- Qwen3-0.6B runs on local hardware
- No data is sent to external APIs
- Model can be air-gapped from network

## Integrity Verification

### Block Chain Structure

Each block contains a hash of the previous block:

```
Block N-1                Block N                 Block N+1
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ prev_hash   │←────────│ prev_hash   │←────────│ prev_hash   │
│ content     │         │ content     │         │ content     │
│ block_hash ─┼────────►│ block_hash ─┼────────►│ block_hash  │
└─────────────┘         └─────────────┘         └─────────────┘
```

### Merkle Tree Verification

Documents are organized in a Merkle tree for efficient verification:

```
                    Root Hash
                   /          \
              Hash AB          Hash CD
             /      \         /      \
         Hash A   Hash B   Hash C   Hash D
           |        |        |        |
         Doc A    Doc B    Doc C    Doc D
```

**Properties:**
- O(log n) proof size
- Any document can be verified without accessing others
- Tampering is detectable at any level

## Threat Model

### Threats Addressed

| Threat | Mitigation |
|--------|------------|
| Data at rest exposure | AES-256-GCM encryption |
| Key compromise | Time-limited session keys |
| Keyword leakage | SHA-256 hashed tags |
| Document tampering | Merkle tree + block hashes |
| Unauthorized access | Master key requirement |
| Memory snooping | Secure key derivation |

### Threats NOT Addressed

| Threat | Reason |
|--------|--------|
| Compromised master key | User responsibility |
| Side-channel attacks | Requires hardware mitigation |
| Malicious LLM responses | LLM safety is separate concern |
| Coercion attacks | Out of scope |

## Attack Resistance

### Brute Force Resistance

- AES-256: 2^256 key space (computationally infeasible)
- PBKDF2 100K iterations: ~100ms per attempt
- At 10K attempts/sec: 10^70 years to exhaust space

### Dictionary Attacks

- Tag hashes include no salt (by design for searchability)
- Mitigation: Encourage complex, multi-word tags
- Alternative: Searchable encryption (future enhancement)

### Replay Attacks

- Session keys include unique nonce
- Keys expire after 30 seconds
- Each encryption uses unique nonce

## Security Recommendations

### Master Key Management

```python
# Generate secure master key
import secrets
master_key = secrets.token_bytes(32)

# Store securely (example: use system keychain)
# NEVER hardcode in source code
# NEVER store in plain text files
```

### Configuration Hardening

```yaml
# Recommended security settings
encryption:
  pbkdf2_iterations: 100000  # Minimum 100K
  
key_generator:
  default_ttl: 30  # Maximum 60 seconds
  
audit:
  enabled: true
  log_level: "INFO"
```

### Deployment Considerations

1. **Storage**: Use encrypted filesystem for additional protection
2. **Network**: Keep system isolated or use TLS
3. **Access**: Implement OS-level access controls
4. **Backup**: Encrypt backups with separate key
5. **Logging**: Enable audit logging for compliance

## Cryptographic Agility

The system is designed for cryptographic agility:

```python
# Encryption algorithm is configurable
config = {
    "algorithm": "AES-256-GCM",  # Current default
    # Future options:
    # "algorithm": "ChaCha20-Poly1305",
    # "algorithm": "AES-256-GCM-SIV",
}
```

## Compliance Considerations

The security model supports compliance with:

- **GDPR**: Data encryption, access controls
- **HIPAA**: Encryption at rest, audit trails
- **SOC 2**: Access logging, key management
- **PCI DSS**: Strong cryptography, key rotation

Note: Compliance requires proper deployment configuration and operational procedures beyond the software itself.

## Security Audit Checklist

- [ ] Master key stored securely
- [ ] PBKDF2 iterations ≥ 100,000
- [ ] Session key TTL ≤ 60 seconds
- [ ] Audit logging enabled
- [ ] Storage directory has restricted permissions
- [ ] No plaintext keys in logs or errors
- [ ] Key rotation procedure documented
- [ ] Backup encryption verified
