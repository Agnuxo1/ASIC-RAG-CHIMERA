# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jun 30 02:55:49 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.88         0.79       1,125,716
ASIC Simulator                  10,000        11.87         1.19       1,374,025
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.22x
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
Tag Lookup                           0.0204       0.0178       0.0445       0.0538       49,047
AND Search (3 tags)                  0.0380       0.0371       0.0471       0.0544       26,287
OR Search (3 tags)                   1.3164       1.2733       1.6015       1.6827          760
Merkle Verification                  5.8695       5.8285       5.9650       6.6282          170
Full Query Pipeline                  6.0133       5.9971       6.1092       6.2432          166
----------------------------------------------------------------------------------------------------
