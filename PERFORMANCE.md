# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jun 25 02:52:05 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.25         0.63       1,378,374
ASIC Simulator                  10,000        11.30         1.13       1,597,184
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.16x
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
Tag Lookup                           0.0176       0.0159       0.0271       0.0420       56,765
AND Search (3 tags)                  0.0389       0.0372       0.0527       0.0630       25,718
OR Search (3 tags)                   1.7004       1.6345       2.0852       2.3829          588
Merkle Verification                  5.2041       5.1867       5.2796       5.4514          192
Full Query Pipeline                  5.3992       5.3901       5.5317       5.7135          185
----------------------------------------------------------------------------------------------------
