# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Sep 12 02:54:05 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.30         0.55       1,587,380
ASIC Simulator                  10,000         8.76         0.88       1,818,271
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.15x
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
Tag Lookup                           0.0197       0.0177       0.0306       0.0485       50,634
AND Search (3 tags)                  0.0363       0.0353       0.0458       0.0538       27,560
OR Search (3 tags)                   1.1504       1.1164       1.4122       1.4704          869
Merkle Verification                  4.3405       4.2964       4.6169       4.7246          230
Full Query Pipeline                  4.4534       4.4122       4.7620       4.9117          225
----------------------------------------------------------------------------------------------------
