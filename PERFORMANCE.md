# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Aug 28 08:46:21 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,391,120
ASIC Simulator                  10,000        10.39         1.04       1,628,441
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.17x
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
Tag Lookup                           0.0179       0.0161       0.0271       0.0421       55,910
AND Search (3 tags)                  0.0373       0.0357       0.0506       0.0586       26,841
OR Search (3 tags)                   1.5108       1.4529       1.8867       2.1047          662
Merkle Verification                  5.2549       5.2480       5.3063       5.3785          190
Full Query Pipeline                  5.4146       5.4022       5.5233       6.1744          185
----------------------------------------------------------------------------------------------------
