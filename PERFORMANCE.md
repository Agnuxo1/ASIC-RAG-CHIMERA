# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jul 19 02:15:22 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.20         0.62       1,388,411
ASIC Simulator                  10,000        10.42         1.04       1,619,101
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.17x
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
Tag Lookup                           0.0192       0.0168       0.0380       0.0470       52,082
AND Search (3 tags)                  0.0384       0.0368       0.0521       0.0610       26,009
OR Search (3 tags)                   1.5839       1.5337       1.9483       2.1283          631
Merkle Verification                  5.2109       5.1866       5.2605       5.4955          192
Full Query Pipeline                  5.3450       5.3569       5.4809       5.5679          187
----------------------------------------------------------------------------------------------------
