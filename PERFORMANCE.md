# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Aug  7 02:28:57 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.86         0.59       1,458,317
ASIC Simulator                  10,000        10.57         1.06       1,663,943
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
Tag Lookup                           0.0203       0.0186       0.0290       0.0496       49,382
AND Search (3 tags)                  0.0391       0.0377       0.0515       0.0600       25,558
OR Search (3 tags)                   1.4027       1.3518       1.7314       1.8645          713
Merkle Verification                  4.6311       4.6216       4.6809       4.7578          216
Full Query Pipeline                  4.7686       4.7797       4.8808       4.9519          210
----------------------------------------------------------------------------------------------------
