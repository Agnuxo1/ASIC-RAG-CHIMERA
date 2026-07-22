# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jul 22 02:12:53 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.05         0.70       1,242,484
ASIC Simulator                  10,000        11.23         1.12       1,442,150
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
Tag Lookup                           0.0187       0.0166       0.0387       0.0417       53,437
AND Search (3 tags)                  0.0365       0.0352       0.0482       0.0550       27,407
OR Search (3 tags)                   1.4806       1.4317       1.8629       2.0830          675
Merkle Verification                  5.2750       5.2650       5.3306       5.3962          190
Full Query Pipeline                  5.3495       5.3479       5.4375       5.5035          187
----------------------------------------------------------------------------------------------------
