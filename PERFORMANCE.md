# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Sep 20 03:06:54 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.51         0.48       1,813,508
ASIC Simulator                  10,000         8.30         0.83       2,077,477
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
Tag Lookup                           0.0157       0.0140       0.0343       0.0385       63,751
AND Search (3 tags)                  0.0305       0.0295       0.0387       0.0441       32,801
OR Search (3 tags)                   1.0358       0.9997       1.2762       1.4196          965
Merkle Verification                  3.7022       3.6976       3.7605       3.8255          270
Full Query Pipeline                  3.8402       3.8306       3.9204       3.9601          260
----------------------------------------------------------------------------------------------------
