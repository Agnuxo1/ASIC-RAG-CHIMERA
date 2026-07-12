# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jul 12 02:18:44 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.05         0.61       1,418,424
ASIC Simulator                  10,000         9.34         0.93       1,744,314
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.23x
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
Tag Lookup                           0.0208       0.0181       0.0440       0.0526       48,082
AND Search (3 tags)                  0.0393       0.0382       0.0499       0.0568       25,421
OR Search (3 tags)                   1.2556       1.2249       1.5188       1.6015          796
Merkle Verification                  4.7165       4.6920       4.9339       5.0811          212
Full Query Pipeline                  4.8657       4.8798       5.1098       5.2086          206
----------------------------------------------------------------------------------------------------
