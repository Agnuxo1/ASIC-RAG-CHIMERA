# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Aug 20 01:00:32 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.26         0.62       1,377,973
ASIC Simulator                  10,000        11.04         1.10       1,579,764
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
Tag Lookup                           0.0178       0.0161       0.0279       0.0415       56,270
AND Search (3 tags)                  0.0377       0.0361       0.0519       0.0593       26,556
OR Search (3 tags)                   1.6614       1.6110       2.1037       2.3984          602
Merkle Verification                  5.2758       5.2596       5.3326       5.6146          190
Full Query Pipeline                  5.4334       5.4263       5.5549       5.7286          184
----------------------------------------------------------------------------------------------------
