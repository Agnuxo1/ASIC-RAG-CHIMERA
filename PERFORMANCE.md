# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jul 20 02:36:16 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.81         0.59       1,468,017
ASIC Simulator                  10,000        10.68         1.07       1,659,459
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.13x
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
Tag Lookup                           0.0206       0.0187       0.0419       0.0518       48,490
AND Search (3 tags)                  0.0404       0.0389       0.0545       0.0612       24,744
OR Search (3 tags)                   1.7882       1.7403       2.2995       2.8040          559
Merkle Verification                  4.6367       4.6248       4.7093       5.0309          216
Full Query Pipeline                  4.8124       4.7997       4.9160       5.0496          208
----------------------------------------------------------------------------------------------------
