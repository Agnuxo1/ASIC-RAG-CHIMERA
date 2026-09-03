# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Sep  3 02:44:05 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.35         0.64       1,361,399
ASIC Simulator                  10,000         9.64         0.96       1,677,302
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
Tag Lookup                           0.0219       0.0192       0.0486       0.0567       45,604
AND Search (3 tags)                  0.0413       0.0401       0.0527       0.0626       24,212
OR Search (3 tags)                   1.2933       1.2380       1.5560       1.8940          773
Merkle Verification                  4.7048       4.6920       4.7838       4.9760          213
Full Query Pipeline                  4.8448       4.8594       4.9604       5.0724          206
----------------------------------------------------------------------------------------------------
