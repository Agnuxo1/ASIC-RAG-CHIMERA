# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jun 22 03:46:34 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,385,715
ASIC Simulator                  10,000        11.62         1.16       1,610,641
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
Tag Lookup                           0.0178       0.0159       0.0266       0.0418       56,106
AND Search (3 tags)                  0.0380       0.0363       0.0512       0.0575       26,338
OR Search (3 tags)                   1.6765       1.6227       2.0446       2.3074          596
Merkle Verification                  5.1684       5.1564       5.2358       5.3242          193
Full Query Pipeline                  5.3562       5.3556       5.4719       5.6032          187
----------------------------------------------------------------------------------------------------
