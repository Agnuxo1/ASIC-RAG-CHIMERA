# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Sep  5 02:41:00 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.36         0.64       1,358,592
ASIC Simulator                  10,000        10.20         1.02       1,657,076
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.22x
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
Tag Lookup                           0.0196       0.0179       0.0327       0.0502       51,027
AND Search (3 tags)                  0.0386       0.0374       0.0495       0.0551       25,916
OR Search (3 tags)                   1.3432       1.2909       1.6695       1.8220          745
Merkle Verification                  4.7502       4.7386       4.8198       5.0522          211
Full Query Pipeline                  4.8677       4.8648       4.9792       5.1999          205
----------------------------------------------------------------------------------------------------
