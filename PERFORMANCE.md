# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jul  8 02:16:13 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,384,187
ASIC Simulator                  10,000        10.83         1.08       1,621,502
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
Tag Lookup                           0.0187       0.0165       0.0348       0.0468       53,587
AND Search (3 tags)                  0.0363       0.0351       0.0481       0.0546       27,517
OR Search (3 tags)                   1.3808       1.3284       1.7043       1.8453          724
Merkle Verification                  5.1916       5.1854       5.2506       5.5753          193
Full Query Pipeline                  5.3928       5.3759       5.5785       5.7086          185
----------------------------------------------------------------------------------------------------
