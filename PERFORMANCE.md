# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jul 28 02:07:48 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.77         0.58       1,477,031
ASIC Simulator                  10,000        10.49         1.05       1,658,510
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.12x
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
Tag Lookup                           0.0205       0.0186       0.0299       0.0519       48,883
AND Search (3 tags)                  0.0387       0.0374       0.0494       0.0554       25,835
OR Search (3 tags)                   1.3573       1.3068       1.6832       1.8620          737
Merkle Verification                  4.6608       4.6568       4.7113       4.7559          215
Full Query Pipeline                  4.7927       4.7693       4.8763       5.1423          209
----------------------------------------------------------------------------------------------------
