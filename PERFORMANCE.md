# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jun 16 03:41:26 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.25         0.62       1,380,024
ASIC Simulator                  10,000        10.79         1.08       1,641,545
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
Tag Lookup                           0.0179       0.0160       0.0281       0.0415       55,825
AND Search (3 tags)                  0.0366       0.0353       0.0475       0.0533       27,348
OR Search (3 tags)                   1.3713       1.3171       1.7093       1.8779          729
Merkle Verification                  5.3009       5.2722       5.3669       5.7310          189
Full Query Pipeline                  5.4273       5.4302       5.5248       5.7442          184
----------------------------------------------------------------------------------------------------
