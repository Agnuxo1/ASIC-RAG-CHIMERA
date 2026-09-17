# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Sep 17 03:08:26 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,385,024
ASIC Simulator                  10,000        10.30         1.03       1,647,100
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.19x
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
Tag Lookup                           0.0197       0.0181       0.0292       0.0496       50,647
AND Search (3 tags)                  0.0398       0.0382       0.0519       0.0592       25,148
OR Search (3 tags)                   1.4022       1.3391       1.7452       2.0463          713
Merkle Verification                  4.7422       4.7303       4.8140       4.9135          211
Full Query Pipeline                  4.8879       4.8722       4.9909       5.0855          205
----------------------------------------------------------------------------------------------------
