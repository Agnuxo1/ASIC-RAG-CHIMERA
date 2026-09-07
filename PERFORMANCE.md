# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Sep  7 02:37:04 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.26         0.63       1,378,232
ASIC Simulator                  10,000        10.15         1.01       1,671,872
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.21x
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
Tag Lookup                           0.0196       0.0180       0.0292       0.0495       50,960
AND Search (3 tags)                  0.0385       0.0373       0.0489       0.0548       25,998
OR Search (3 tags)                   1.3380       1.2882       1.6473       1.7477          747
Merkle Verification                  4.7186       4.7168       4.7758       4.8613          212
Full Query Pipeline                  4.8739       4.8349       4.9734       6.0433          205
----------------------------------------------------------------------------------------------------
