# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Aug  6 02:09:27 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.34         0.63       1,361,622
ASIC Simulator                  10,000        10.98         1.10       1,552,077
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
Tag Lookup                           0.0183       0.0164       0.0365       0.0413       54,517
AND Search (3 tags)                  0.0367       0.0355       0.0481       0.0574       27,223
OR Search (3 tags)                   1.4032       1.3442       1.7439       2.0033          713
Merkle Verification                  5.3247       5.3067       5.3922       5.5948          188
Full Query Pipeline                  5.5247       5.5112       5.6583       6.4514          181
----------------------------------------------------------------------------------------------------
