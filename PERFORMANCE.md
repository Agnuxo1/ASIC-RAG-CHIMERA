# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jul 13 02:25:07 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.78         0.59       1,474,318
ASIC Simulator                  10,000        10.32         1.03       1,646,825
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.12x
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
Tag Lookup                           0.0208       0.0186       0.0370       0.0538       47,996
AND Search (3 tags)                  0.0403       0.0391       0.0523       0.0592       24,796
OR Search (3 tags)                   1.5674       1.5213       1.9619       2.1270          638
Merkle Verification                  4.6131       4.6069       4.6754       4.7530          217
Full Query Pipeline                  4.8035       4.7549       4.8959       6.0463          208
----------------------------------------------------------------------------------------------------
