# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jun 17 03:40:06 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.29         0.62       1,372,378
ASIC Simulator                  10,000        11.13         1.11       1,623,196
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.18x
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
Tag Lookup                           0.0180       0.0158       0.0307       0.0432       55,651
AND Search (3 tags)                  0.0376       0.0363       0.0497       0.0583       26,626
OR Search (3 tags)                   1.5150       1.4644       1.8685       2.0069          660
Merkle Verification                  5.2193       5.1925       5.2798       5.5650          192
Full Query Pipeline                  5.3710       5.3681       5.4646       5.6368          186
----------------------------------------------------------------------------------------------------
