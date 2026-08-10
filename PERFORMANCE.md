# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Aug 10 01:23:19 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.71         0.58       1,489,896
ASIC Simulator                  10,000        10.38         1.04       1,660,606
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
Tag Lookup                           0.0206       0.0186       0.0348       0.0491       48,534
AND Search (3 tags)                  0.0391       0.0377       0.0502       0.0579       25,580
OR Search (3 tags)                   1.3683       1.3106       1.7136       2.0143          731
Merkle Verification                  4.8488       4.8412       4.8993       5.0340          206
Full Query Pipeline                  4.9954       4.9882       5.0905       5.2506          200
----------------------------------------------------------------------------------------------------
