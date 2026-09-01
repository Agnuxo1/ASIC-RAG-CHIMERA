# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Sep  1 03:17:55 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.25         0.62       1,378,803
ASIC Simulator                  10,000        10.34         1.03       1,659,946
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.20x
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
Tag Lookup                           0.0200       0.0180       0.0392       0.0495       49,911
AND Search (3 tags)                  0.0403       0.0390       0.0521       0.0597       24,815
OR Search (3 tags)                   1.7091       1.6433       2.1447       2.2842          585
Merkle Verification                  4.7178       4.7036       4.7986       5.0255          212
Full Query Pipeline                  4.8327       4.8292       4.9267       5.0382          207
----------------------------------------------------------------------------------------------------
