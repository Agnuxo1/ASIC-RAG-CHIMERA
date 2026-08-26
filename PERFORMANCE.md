# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Aug 26 01:03:35 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.07         0.61       1,415,315
ASIC Simulator                  10,000        10.26         1.03       1,661,438
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
Tag Lookup                           0.0196       0.0181       0.0290       0.0494       51,134
AND Search (3 tags)                  0.0406       0.0390       0.0536       0.0623       24,603
OR Search (3 tags)                   1.6899       1.6366       2.0962       2.2605          592
Merkle Verification                  4.6973       4.6893       4.7730       4.8888          213
Full Query Pipeline                  4.8333       4.8240       4.9208       5.5747          207
----------------------------------------------------------------------------------------------------
