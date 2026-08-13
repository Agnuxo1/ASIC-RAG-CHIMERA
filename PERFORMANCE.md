# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Aug 13 01:30:46 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.90         0.60       1,449,969
ASIC Simulator                  10,000        10.68         1.07       1,607,766
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.11x
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
Tag Lookup                           0.0208       0.0187       0.0433       0.0488       48,072
AND Search (3 tags)                  0.0405       0.0392       0.0534       0.0613       24,694
OR Search (3 tags)                   1.7010       1.6383       2.0957       2.2703          588
Merkle Verification                  4.6166       4.6081       4.6750       4.8814          217
Full Query Pipeline                  4.7569       4.7596       4.8596       4.9321          210
----------------------------------------------------------------------------------------------------
