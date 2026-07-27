# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jul 27 02:33:20 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,385,217
ASIC Simulator                  10,000        10.61         1.06       1,614,860
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.17x
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
Tag Lookup                           0.0186       0.0165       0.0348       0.0462       53,668
AND Search (3 tags)                  0.0381       0.0365       0.0511       0.0617       26,271
OR Search (3 tags)                   1.7411       1.6765       2.1795       2.5504          574
Merkle Verification                  5.2734       5.2683       5.3576       5.5840          190
Full Query Pipeline                  5.4065       5.3910       5.5103       5.6766          185
----------------------------------------------------------------------------------------------------
