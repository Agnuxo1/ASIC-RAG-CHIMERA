# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Sep 18 02:56:28 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.61         0.48       1,783,342
ASIC Simulator                  10,000         8.06         0.81       2,123,208
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
Tag Lookup                           0.0154       0.0141       0.0219       0.0390       64,780
AND Search (3 tags)                  0.0312       0.0302       0.0396       0.0459       32,098
OR Search (3 tags)                   1.1102       1.0729       1.4139       1.6277          901
Merkle Verification                  3.6930       3.6916       3.7427       3.7925          271
Full Query Pipeline                  3.8189       3.8202       3.9160       3.9613          262
----------------------------------------------------------------------------------------------------
