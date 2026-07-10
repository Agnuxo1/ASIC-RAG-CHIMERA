# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jul 10 02:37:16 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.27         0.46       1,898,834
ASIC Simulator                  10,000         8.17         0.82       2,140,451
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
Tag Lookup                           0.0158       0.0145       0.0232       0.0400       63,255
AND Search (3 tags)                  0.0307       0.0296       0.0400       0.0473       32,525
OR Search (3 tags)                   1.0511       1.0063       1.2988       1.4009          951
Merkle Verification                  3.6448       3.6165       3.7116       4.7822          274
Full Query Pipeline                  3.7040       3.6871       3.8020       4.1666          270
----------------------------------------------------------------------------------------------------
