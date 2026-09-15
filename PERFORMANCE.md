# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Sep 15 03:08:42 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.13         0.61       1,402,335
ASIC Simulator                  10,000        10.65         1.07       1,597,876
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.14x
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
Tag Lookup                           0.0184       0.0161       0.0353       0.0462       54,490
AND Search (3 tags)                  0.0381       0.0364       0.0521       0.0611       26,240
OR Search (3 tags)                   1.5202       1.4036       2.1272       2.4788          658
Merkle Verification                  5.3291       5.3305       5.4189       5.5979          188
Full Query Pipeline                  5.5537       5.5697       5.7084       5.8691          180
----------------------------------------------------------------------------------------------------
