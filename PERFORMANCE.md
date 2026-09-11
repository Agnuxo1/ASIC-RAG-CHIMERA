# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Sep 11 02:44:10 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.94         0.59       1,440,339
ASIC Simulator                  10,000        10.12         1.01       1,662,249
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.15x
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
Tag Lookup                           0.0198       0.0181       0.0280       0.0492       50,604
AND Search (3 tags)                  0.0392       0.0381       0.0497       0.0556       25,499
OR Search (3 tags)                   1.5562       1.4970       1.9104       2.0833          643
Merkle Verification                  4.7406       4.7308       4.8121       4.9425          211
Full Query Pipeline                  4.8482       4.8576       4.9624       5.0598          206
----------------------------------------------------------------------------------------------------
