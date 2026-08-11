# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Aug 11 01:20:57 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.79         0.59       1,472,663
ASIC Simulator                  10,000        10.68         1.07       1,593,643
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.08x
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
Tag Lookup                           0.0205       0.0185       0.0336       0.0529       48,761
AND Search (3 tags)                  0.0400       0.0387       0.0522       0.0598       24,982
OR Search (3 tags)                   1.5203       1.4725       1.9107       2.1361          658
Merkle Verification                  4.6480       4.6261       4.7141       5.1846          215
Full Query Pipeline                  4.8196       4.8097       4.9224       5.0254          207
----------------------------------------------------------------------------------------------------
