# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Aug  9 01:21:00 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.12         0.61       1,404,369
ASIC Simulator                  10,000        10.57         1.06       1,613,589
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
Tag Lookup                           0.0184       0.0165       0.0312       0.0470       54,336
AND Search (3 tags)                  0.0364       0.0350       0.0481       0.0576       27,468
OR Search (3 tags)                   1.4157       1.3584       1.7479       1.9405          706
Merkle Verification                  5.3018       5.2955       5.3613       5.4499          189
Full Query Pipeline                  5.4684       5.4507       5.5920       5.7793          183
----------------------------------------------------------------------------------------------------
