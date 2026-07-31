# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jul 31 02:25:57 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.18         0.61       1,392,639
ASIC Simulator                  10,000        10.61         1.06       1,608,877
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
Tag Lookup                           0.0184       0.0165       0.0376       0.0416       54,344
AND Search (3 tags)                  0.0385       0.0368       0.0525       0.0586       25,954
OR Search (3 tags)                   1.7569       1.6796       2.1384       2.3679          569
Merkle Verification                  5.2568       5.2465       5.3224       5.4409          190
Full Query Pipeline                  5.4169       5.4127       5.5226       5.6652          185
----------------------------------------------------------------------------------------------------
