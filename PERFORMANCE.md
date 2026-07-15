# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jul 15 02:01:44 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.25         0.62       1,378,665
ASIC Simulator                  10,000        10.95         1.10       1,555,567
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.13x
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
Tag Lookup                           0.0188       0.0168       0.0309       0.0420       53,133
AND Search (3 tags)                  0.0389       0.0370       0.0537       0.0624       25,676
OR Search (3 tags)                   1.8875       1.8237       2.3263       2.7270          530
Merkle Verification                  5.2429       5.2292       5.3493       5.5552          191
Full Query Pipeline                  5.4323       5.3966       5.6871       6.1311          184
----------------------------------------------------------------------------------------------------
