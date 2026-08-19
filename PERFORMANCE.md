# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Aug 19 01:00:25 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.77         0.64       1,286,734
ASIC Simulator                  10,000        10.55         1.05       1,602,567
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.25x
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
Tag Lookup                           0.0187       0.0166       0.0370       0.0444       53,449
AND Search (3 tags)                  0.0377       0.0362       0.0518       0.0595       26,521
OR Search (3 tags)                   1.7383       1.6795       2.1448       2.3887          575
Merkle Verification                  5.1792       5.1642       5.2512       5.4466          193
Full Query Pipeline                  5.3825       5.3833       5.4987       5.6676          186
----------------------------------------------------------------------------------------------------
