# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Aug  8 01:16:33 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.15         0.61       1,398,444
ASIC Simulator                  10,000        10.46         1.05       1,584,793
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
Tag Lookup                           0.0182       0.0164       0.0314       0.0412       54,890
AND Search (3 tags)                  0.0354       0.0340       0.0467       0.0531       28,263
OR Search (3 tags)                   1.2966       1.2501       1.5930       1.6372          771
Merkle Verification                  5.2617       5.2475       5.3246       5.6789          190
Full Query Pipeline                  5.4276       5.4276       5.5248       5.6137          184
----------------------------------------------------------------------------------------------------
