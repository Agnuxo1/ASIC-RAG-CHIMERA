# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jul 25 02:15:02 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.29         0.46       1,889,632
ASIC Simulator                  10,000         7.94         0.79       2,158,303
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
Tag Lookup                           0.0165       0.0147       0.0259       0.0398       60,749
AND Search (3 tags)                  0.0305       0.0296       0.0392       0.0435       32,812
OR Search (3 tags)                   1.0557       1.0106       1.3109       1.4941          947
Merkle Verification                  3.5850       3.5802       3.6312       3.6616          279
Full Query Pipeline                  3.6636       3.6630       3.7613       3.8279          273
----------------------------------------------------------------------------------------------------
