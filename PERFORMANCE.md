# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jun 14 10:34:24 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.65         0.76       1,156,325
ASIC Simulator                  10,000        12.32         1.23       1,338,129
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
Tag Lookup                           0.0204       0.0177       0.0358       0.0529       48,947
AND Search (3 tags)                  0.0381       0.0370       0.0483       0.0563       26,213
OR Search (3 tags)                   1.3560       1.3039       1.6496       1.8166          737
Merkle Verification                  5.6676       5.6307       5.8008       5.9360          176
Full Query Pipeline                  5.8618       5.8602       5.9931       6.1434          171
----------------------------------------------------------------------------------------------------
