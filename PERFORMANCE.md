# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jun 15 03:45:53 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.92         0.70       1,262,105
ASIC Simulator                  10,000        11.73         1.17       1,443,041
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.14x
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
Tag Lookup                           0.0208       0.0177       0.0439       0.0524       48,108
AND Search (3 tags)                  0.0379       0.0370       0.0474       0.0532       26,371
OR Search (3 tags)                   1.3052       1.2564       1.5825       1.6623          766
Merkle Verification                  6.1519       6.1304       6.2645       6.3822          163
Full Query Pipeline                  6.3627       6.3608       6.5013       6.6168          157
----------------------------------------------------------------------------------------------------
