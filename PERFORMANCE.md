# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jun 23 02:51:24 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.31         0.71       1,202,788
ASIC Simulator                  10,000        10.99         1.10       1,617,999
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.35x
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
Tag Lookup                           0.0180       0.0159       0.0278       0.0428       55,614
AND Search (3 tags)                  0.0381       0.0363       0.0511       0.0611       26,275
OR Search (3 tags)                   1.5850       1.5303       1.9333       2.1393          631
Merkle Verification                  5.1607       5.1477       5.2253       5.3341          194
Full Query Pipeline                  5.3417       5.3445       5.4489       5.5516          187
----------------------------------------------------------------------------------------------------
