# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Aug 22 01:00:17 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,386,842
ASIC Simulator                  10,000        10.38         1.04       1,579,442
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
Tag Lookup                           0.0180       0.0160       0.0360       0.0412       55,442
AND Search (3 tags)                  0.0367       0.0351       0.0482       0.0556       27,267
OR Search (3 tags)                   1.4600       1.4028       1.7975       1.9885          685
Merkle Verification                  5.2513       5.2433       5.3049       5.3643          190
Full Query Pipeline                  5.4319       5.4300       5.5626       5.7160          184
----------------------------------------------------------------------------------------------------
