# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Sep 21 03:03:34 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         8.40         0.72       1,190,282
ASIC Simulator                  10,000        10.49         1.05       1,590,662
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.34x
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
Tag Lookup                           0.0179       0.0160       0.0324       0.0411       55,732
AND Search (3 tags)                  0.0355       0.0342       0.0466       0.0516       28,180
OR Search (3 tags)                   1.3699       1.3141       1.6815       1.8539          730
Merkle Verification                  5.2488       5.2393       5.3288       5.4174          191
Full Query Pipeline                  5.4726       5.4649       5.5721       5.8662          183
----------------------------------------------------------------------------------------------------
