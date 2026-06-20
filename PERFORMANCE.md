# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jun 20 02:57:02 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.47         0.75       1,181,291
ASIC Simulator                  10,000        12.03         1.20       1,347,002
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
Tag Lookup                           0.0205       0.0178       0.0271       0.0508       48,675
AND Search (3 tags)                  0.0386       0.0376       0.0484       0.0554       25,900
OR Search (3 tags)                   1.3461       1.2874       1.6380       1.9538          743
Merkle Verification                  6.1151       6.1035       6.2354       6.3188          164
Full Query Pipeline                  6.2633       6.2679       6.4060       6.5010          160
----------------------------------------------------------------------------------------------------
