# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Aug 25 01:02:07 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,387,692
ASIC Simulator                  10,000        10.66         1.07       1,604,435
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.16x
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
Tag Lookup                           0.0179       0.0162       0.0277       0.0416       55,953
AND Search (3 tags)                  0.0388       0.0372       0.0527       0.0603       25,756
OR Search (3 tags)                   1.7653       1.7094       2.2207       2.4767          566
Merkle Verification                  5.3284       5.3157       5.3984       5.6736          188
Full Query Pipeline                  5.5311       5.5111       5.6512       6.4410          181
----------------------------------------------------------------------------------------------------
