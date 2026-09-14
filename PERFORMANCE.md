# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Sep 14 03:06:34 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.40         0.47       1,853,417
ASIC Simulator                  10,000         7.98         0.80       2,122,802
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.15x
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
Tag Lookup                           0.0157       0.0143       0.0263       0.0392       63,565
AND Search (3 tags)                  0.0337       0.0324       0.0459       0.0527       29,640
OR Search (3 tags)                   1.5874       1.5317       1.9606       2.1703          630
Merkle Verification                  3.6822       3.6747       3.7319       3.8388          272
Full Query Pipeline                  3.8198       3.8107       3.9056       3.9744          262
----------------------------------------------------------------------------------------------------
