# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jun 19 03:54:32 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.01         0.70       1,248,610
ASIC Simulator                  10,000        11.66         1.17       1,450,677
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
Tag Lookup                           0.0210       0.0179       0.0328       0.0483       47,717
AND Search (3 tags)                  0.0389       0.0377       0.0497       0.0552       25,681
OR Search (3 tags)                   1.3216       1.2712       1.5990       1.6929          757
Merkle Verification                  5.4826       5.4646       5.5631       5.6609          182
Full Query Pipeline                  5.6365       5.6423       5.7515       5.9161          177
----------------------------------------------------------------------------------------------------
