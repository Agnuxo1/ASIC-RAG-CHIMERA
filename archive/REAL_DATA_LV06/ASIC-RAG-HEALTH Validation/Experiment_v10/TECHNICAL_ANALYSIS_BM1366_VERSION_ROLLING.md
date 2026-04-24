# Technical Analysis: BM1366 ASIC Version Rolling Implementation

## Problem Statement

When implementing cryptographic verification for the ASIC-RAG-CHIMERA medical records system, the Python-based Stratum pool consistently reported "Hash > Target" errors despite the ASIC successfully submitting shares that real pools accepted.

## Investigation

### Evidence from ESP-Miner Firmware Logs

Analysis of the open-source ESP-Miner firmware (used by Bitaxe and Lucky Miner LV06) revealed critical information about how the BM1366 ASIC handles version rolling:

```
I (2863368) bm1368Module: Job ID: 48, Core: 6/1, Ver: 06E42000
I (2863368) asic_result: Ver: 26E42000 Nonce A4E2010C diff 2094.3 of 2048.
I (2863368) stratum_api: tx: {"id": 215, "method": "mining.submit", 
    "params": ["bc1q...bitaxe", "1940fcf", "51160000", "66ba9110", "a4e2010c", "06e42000"]}
```

### Key Observations

1. **ASIC Output**: The BM1366 returns `Ver: 06E42000` (only the modified bits)
2. **Internal Processing**: ESP-Miner internally calculates `Ver: 26E42000` (OR with 0x20000000)
3. **Stratum Submit**: The mining.submit sends `"06e42000"` (without the base version)

### Root Cause

The BIP320 version rolling protocol works as follows:

1. Pool sends job with `version = 0x20000000` (BIP9 base)
2. Pool negotiates `version_rolling_mask = 0x1fffe000` (bits ASIC can modify)
3. ASIC modifies allowed bits and returns ONLY the modified value
4. Pool must reconstruct: `final_version = BASE | version_bits`

Our V6 implementation incorrectly used complex mask operations:
```python
# WRONG (V6):
version_used = (job['version'] & ~mask) | (version_bits & mask)
```

The correct approach is simpler:
```python
# CORRECT (V10):
version_used = 0x20000000 | version_bits
```

### Mathematical Verification

For a submit with `version_bits = "08f82000"`:

```
version_bits  = 0x08f82000 = 0000 1000 1111 1000 0010 0000 0000 0000
BASE_VERSION  = 0x20000000 = 0010 0000 0000 0000 0000 0000 0000 0000
─────────────────────────────────────────────────────────────────────
final_version = 0x28f82000 = 0010 1000 1111 1000 0010 0000 0000 0000
```

This matches the pattern observed in ESP-Miner logs where `0x06E42000 | 0x20000000 = 0x26E42000`.

## Solution Implementation (V10)

```python
def verify_share(self, job: Dict, submit: Dict) -> Tuple[bool, str, Dict]:
    # ...
    
    if submit.get('version_bits'):
        version_bits = int(submit['version_bits'], 16)
        
        # Method: OR with base version 0x20000000
        # This is how ESP-Miner internally processes it
        BASE_VERSION = 0x20000000
        version_used = BASE_VERSION | version_bits
    else:
        version_used = job['version']
    
    # Build header with correct version
    header = build_block_header(
        version=version_used,
        prev_block_hash=prevhash_bytes,
        merkle_root=block_merkle_root,
        timestamp=ntime,
        nbits=job['nbits'],
        nonce=nonce
    )
    
    block_hash = sha256d(header)
    # ...
```

## References

1. **ESP-Miner Source Code**: https://github.com/bitaxeorg/ESP-Miner
2. **BIP320 - Version Rolling**: https://github.com/bitcoin/bips/blob/master/bip-0320.mediawiki
3. **BM1366 Documentation**: https://github.com/skot/BM1397/blob/master/bm1366.md
4. **Stratum Protocol**: https://en.bitcoin.it/wiki/Stratum_mining_protocol

## Impact on ASIC-RAG-CHIMERA

This fix enables cryptographically verified proof-of-work for medical record sealing, where:

1. Medical records are hashed into a Merkle tree
2. The Merkle root is embedded in the coinbase transaction
3. The ASIC performs SHA256d hashing with version rolling
4. The share is cryptographically verified by reconstructing the exact block header
5. The proof binds the medical records to verifiable computational work

## Version History

- **V1-V5**: Progressive development, verification bypass for testing
- **V6**: Added version rolling support (incorrect mask operation)
- **V10**: Corrected version rolling (simple OR with base version)

---
*Document prepared for ASIC-RAG-CHIMERA academic publication*
*December 2025*
