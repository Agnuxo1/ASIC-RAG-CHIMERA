# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jul 16 02:11:52 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.61       1,390,787
ASIC Simulator                  10,000        10.36         1.04       1,609,768
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.16x
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
Tag Lookup                           0.0188       0.0166       0.0388       0.0435       53,301
AND Search (3 tags)                  0.0386       0.0371       0.0531       0.0628       25,886
OR Search (3 tags)                   1.5716       1.4979       1.9530       2.3722          636
Merkle Verification                  5.2647       5.2731       5.3497       5.5460          190
Full Query Pipeline                  5.4503       5.4122       5.5557       7.8199          183
----------------------------------------------------------------------------------------------------
