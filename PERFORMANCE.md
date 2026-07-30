# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jul 30 02:00:44 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.72         0.58       1,488,747
ASIC Simulator                  10,000        10.51         1.05       1,657,533
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.11x
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
Tag Lookup                           0.0210       0.0187       0.0335       0.0537       47,562
AND Search (3 tags)                  0.0409       0.0387       0.0544       0.0914       24,449
OR Search (3 tags)                   1.5541       1.4608       2.0189       2.8876          643
Merkle Verification                  4.6052       4.5831       4.6706       4.9935          217
Full Query Pipeline                  4.7410       4.7373       4.8506       4.9401          211
----------------------------------------------------------------------------------------------------
