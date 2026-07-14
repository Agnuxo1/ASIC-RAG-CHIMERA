# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jul 14 02:05:06 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,386,989
ASIC Simulator                  10,000        10.18         1.02       1,631,306
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.18x
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
Tag Lookup                           0.0221       0.0196       0.0303       0.0546       45,177
AND Search (3 tags)                  0.0464       0.0427       0.0713       0.0842       21,550
OR Search (3 tags)                   1.3819       1.3348       1.6612       1.9498          724
Merkle Verification                  4.9232       4.9111       5.0083       5.3039          203
Full Query Pipeline                  5.0936       5.0838       5.2008       5.2770          196
----------------------------------------------------------------------------------------------------
