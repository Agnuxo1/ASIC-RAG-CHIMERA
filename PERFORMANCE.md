# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Sep 19 02:54:29 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         4.28         0.37       2,335,806
ASIC Simulator                  10,000         6.26         0.63       2,542,971
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.09x
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
Tag Lookup                           0.0137       0.0128       0.0178       0.0358       73,010
AND Search (3 tags)                  0.0276       0.0262       0.0371       0.0462       36,174
OR Search (3 tags)                   0.7679       0.7369       0.9556       1.1596        1,302
Merkle Verification                  2.7645       2.7276       2.8774       3.4699          362
Full Query Pipeline                  2.8333       2.8197       2.9803       3.0393          353
----------------------------------------------------------------------------------------------------
