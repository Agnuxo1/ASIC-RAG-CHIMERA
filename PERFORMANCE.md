# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Sep  2 02:36:05 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,384,501
ASIC Simulator                  10,000        10.55         1.05       1,599,648
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
Tag Lookup                           0.0182       0.0161       0.0351       0.0445       54,977
AND Search (3 tags)                  0.0360       0.0346       0.0476       0.0544       27,771
OR Search (3 tags)                   1.3633       1.3167       1.6793       1.7381          734
Merkle Verification                  5.2279       5.2169       5.3004       5.3899          191
Full Query Pipeline                  5.3229       5.3367       5.4442       5.5396          188
----------------------------------------------------------------------------------------------------
