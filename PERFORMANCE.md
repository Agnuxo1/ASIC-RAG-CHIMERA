# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Aug  1 02:27:24 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.79         0.59       1,472,262
ASIC Simulator                  10,000        10.18         1.02       1,669,872
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
Tag Lookup                           0.0209       0.0187       0.0442       0.0493       47,906
AND Search (3 tags)                  0.0394       0.0383       0.0507       0.0550       25,359
OR Search (3 tags)                   1.3879       1.3383       1.7129       1.8646          720
Merkle Verification                  4.5491       4.5443       4.6078       4.6738          220
Full Query Pipeline                  4.7083       4.7039       4.8175       4.9350          212
----------------------------------------------------------------------------------------------------
