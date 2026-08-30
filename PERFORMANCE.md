# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Aug 30 03:17:57 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.34         0.46       1,871,324
ASIC Simulator                  10,000         8.72         0.87       2,139,023
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.14x
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
Tag Lookup                           0.0156       0.0142       0.0234       0.0385       64,037
AND Search (3 tags)                  0.0319       0.0307       0.0422       0.0491       31,307
OR Search (3 tags)                   1.3353       1.3117       1.7157       1.8901          749
Merkle Verification                  3.6460       3.6141       3.6901       5.0289          274
Full Query Pipeline                  3.7513       3.7403       3.8413       3.9684          267
----------------------------------------------------------------------------------------------------
