# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jul  1 03:02:30 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.18         0.62       1,391,869
ASIC Simulator                  10,000        11.18         1.12       1,609,051
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
Tag Lookup                           0.0185       0.0165       0.0330       0.0443       54,116
AND Search (3 tags)                  0.0371       0.0358       0.0492       0.0566       26,921
OR Search (3 tags)                   1.4189       1.3628       1.7680       1.9889          705
Merkle Verification                  5.1990       5.1883       5.2650       5.3447          192
Full Query Pipeline                  5.3606       5.3519       5.4509       5.5250          187
----------------------------------------------------------------------------------------------------
