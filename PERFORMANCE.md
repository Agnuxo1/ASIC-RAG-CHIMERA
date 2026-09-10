# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Sep 10 02:50:26 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,386,739
ASIC Simulator                  10,000        10.17         1.02       1,655,775
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.19x
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
Tag Lookup                           0.0193       0.0179       0.0286       0.0490       51,689
AND Search (3 tags)                  0.0391       0.0376       0.0495       0.0554       25,604
OR Search (3 tags)                   1.3676       1.3103       1.6856       2.0522          731
Merkle Verification                  4.7800       4.7736       4.8409       4.9453          209
Full Query Pipeline                  4.9432       4.9374       5.0544       5.1543          202
----------------------------------------------------------------------------------------------------
