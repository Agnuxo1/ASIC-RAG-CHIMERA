# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jun 18 03:32:56 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.80         0.59       1,469,810
ASIC Simulator                  10,000        11.20         1.12       1,643,903
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
Tag Lookup                           0.0198       0.0178       0.0334       0.0494       50,591
AND Search (3 tags)                  0.0410       0.0397       0.0535       0.0609       24,377
OR Search (3 tags)                   1.6541       1.6214       2.0838       2.3409          605
Merkle Verification                  4.6230       4.6093       4.6959       4.8131          216
Full Query Pipeline                  4.9114       4.9028       5.0469       5.2360          204
----------------------------------------------------------------------------------------------------
