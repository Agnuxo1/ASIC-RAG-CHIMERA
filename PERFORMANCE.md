# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Aug 29 05:23:51 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,391,030
ASIC Simulator                  10,000        10.74         1.07       1,605,171
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
Tag Lookup                           0.0183       0.0161       0.0314       0.0420       54,547
AND Search (3 tags)                  0.0380       0.0365       0.0508       0.0582       26,291
OR Search (3 tags)                   1.8856       1.8003       2.4081       3.0997          530
Merkle Verification                  5.3590       5.3456       5.4852       5.6293          187
Full Query Pipeline                  5.4868       5.4702       5.6383       5.9323          182
----------------------------------------------------------------------------------------------------
