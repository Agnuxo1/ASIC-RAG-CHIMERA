# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Sep  9 02:48:46 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.68         0.66       1,301,839
ASIC Simulator                  10,000        10.30         1.03       1,598,612
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.23x
Setting up benchmark with 10000 documents...
  Index size: 105 tags
  Merkle tree: 10000 leaves
Running tag lookup benchmark...
Running AND search benchmark...
Running OR search benchmark...
Running Merkle verification benchmark...
Running full query benchmark...

====================================================================================================
SEARCH LATENCY BENCHMARK RESULTS
====================================================================================================
Operation                         Mean (ms)     P50 (ms)     P95 (ms)     P99 (ms)          QPS
----------------------------------------------------------------------------------------------------
Tag Lookup                           0.0177       0.0161       0.0290       0.0421       56,487
AND Search (3 tags)                  0.0374       0.0360       0.0491       0.0575       26,772
OR Search (3 tags)                   1.5459       1.4714       1.9505       2.3039          647
Merkle Verification                  5.3012       5.2733       5.4000       6.2734          189
Full Query Pipeline                  5.5096       5.4899       5.7232       6.7112          182
----------------------------------------------------------------------------------------------------
