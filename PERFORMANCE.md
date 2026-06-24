# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jun 24 02:51:42 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.32         0.46       1,878,802
ASIC Simulator                  10,000         8.53         0.85       2,128,305
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
Tag Lookup                           0.0157       0.0138       0.0264       0.0386       63,817
AND Search (3 tags)                  0.0321       0.0310       0.0423       0.0473       31,161
OR Search (3 tags)                   1.4375       1.3878       1.7910       1.9370          696
Merkle Verification                  3.6569       3.6517       3.7119       3.7994          273
Full Query Pipeline                  3.7912       3.7801       3.8722       3.9265          264
----------------------------------------------------------------------------------------------------
