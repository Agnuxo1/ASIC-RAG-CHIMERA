# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Sep  6 02:38:11 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.42         0.64       1,347,461
ASIC Simulator                  10,000        10.33         1.03       1,613,663
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.20x
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
Tag Lookup                           0.0183       0.0161       0.0389       0.0444       54,588
AND Search (3 tags)                  0.0366       0.0355       0.0480       0.0540       27,294
OR Search (3 tags)                   1.5006       1.4342       1.8658       2.0982          666
Merkle Verification                  5.2874       5.2824       5.3382       5.4149          189
Full Query Pipeline                  5.4330       5.4269       5.5461       5.6426          184
----------------------------------------------------------------------------------------------------
