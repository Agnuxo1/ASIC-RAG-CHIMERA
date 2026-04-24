# ASIC-RAG-CHIMERA V10 - Summary of Changes

## Identified Problem

Cryptographic verification failed because the V6 code incorrectly reconstructed the `version` field of the block header.

## Evidence from ESP-Miner (Bitaxe/LV06)

```
bm1368Module: Ver: 06E42000   <- ASIC returns this
asic_result: Ver: 26E42000    <- After OR: 0x20000000 | 0x06E42000
Submit sends: "06e42000"      <- What we receive in mining.submit
```

## Key Change

### V6 (Incorrect):
```python
version_used = (job['version'] & ~mask) | (version_bits & mask)
```

### V10 (Correct):
```python
BASE_VERSION = 0x20000000
if version_bits & BASE_VERSION:
    version_used = version_bits  # Already includes base
else:
    version_used = BASE_VERSION | version_bits  # Needs OR
```

## Example with your LV06

Your ASIC sends: `version_bits = "08f82000"`

```
0x08f82000 = 0000 1000 1111 1000 0010 0000 0000 0000
0x20000000 = 0010 0000 0000 0000 0000 0000 0000 0000
─────────────────────────────────────────────────────
0x28f82000 = 0010 1000 1111 1000 0010 0000 0000 0000  ← Final Version
```

## To Test

```bash
python3 chimera_medical_handoff_v10.py
```

In the logs you should see:
```
[DEBUG] Version (OR): 08f82000 | 0x20000000 -> 28f82000
```

And if verification works:
```
[SEAL] Cryptographic verification: PASSED
```

## Delivered Files

1. `chimera_medical_handoff_v10.py` - Corrected code
2. `TECHNICAL_ANALYSIS_BM1366_VERSION_ROLLING.md` - Technical analysis for the paper

## References Used

- ESP-Miner logs: https://github.com/bitaxeorg/ESP-Miner/issues/286
- BIP320 Version Rolling
- BM1366 documentation

---
*Let me know if this works with your LV06!*
