# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Sep  8 02:50:23 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.11         0.44       1,958,230
ASIC Simulator                  10,000         8.11         0.81       2,050,445
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.05x
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
Tag Lookup                           0.0165       0.0142       0.0341       0.0403       60,734
AND Search (3 tags)                  0.0292       0.0281       0.0372       0.0507       34,271
OR Search (3 tags)                   0.9874       0.9622       1.1729       1.2043        1,013
Merkle Verification                  3.3655       3.3640       3.4017       3.4740          297
Full Query Pipeline                  3.5181       3.5021       3.6054       3.9186          284
----------------------------------------------------------------------------------------------------
