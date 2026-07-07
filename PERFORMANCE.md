# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jul  7 02:42:59 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.31         0.63       1,368,642
ASIC Simulator                  10,000        11.05         1.11       1,607,212
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.17x
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
Tag Lookup                           0.0187       0.0167       0.0333       0.0431       53,407
AND Search (3 tags)                  0.0390       0.0373       0.0537       0.0613       25,666
OR Search (3 tags)                   1.8513       1.7760       2.2380       2.5427          540
Merkle Verification                  5.2936       5.2875       5.4224       5.5200          189
Full Query Pipeline                  5.3684       5.3615       5.4821       5.5834          186
----------------------------------------------------------------------------------------------------
