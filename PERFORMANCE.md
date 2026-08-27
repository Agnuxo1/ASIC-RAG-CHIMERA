# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Aug 27 06:58:06 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.42         0.64       1,348,340
ASIC Simulator                  10,000        10.42         1.04       1,610,403
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
Tag Lookup                           0.0186       0.0161       0.0396       0.0462       53,868
AND Search (3 tags)                  0.0371       0.0357       0.0494       0.0572       26,980
OR Search (3 tags)                   1.5909       1.5268       1.9660       2.3661          629
Merkle Verification                  5.2190       5.2042       5.2935       5.4491          192
Full Query Pipeline                  5.3252       5.3308       5.4403       5.5473          188
----------------------------------------------------------------------------------------------------
