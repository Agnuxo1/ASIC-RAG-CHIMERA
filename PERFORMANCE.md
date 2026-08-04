# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Aug  4 02:06:47 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.73         0.58       1,486,374
ASIC Simulator                  10,000        10.37         1.04       1,689,287
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
Tag Lookup                           0.0208       0.0187       0.0299       0.0515       48,022
AND Search (3 tags)                  0.0393       0.0382       0.0499       0.0555       25,452
OR Search (3 tags)                   1.3837       1.3389       1.7116       1.8352          723
Merkle Verification                  4.5839       4.5749       4.6433       4.7855          218
Full Query Pipeline                  4.7298       4.7202       4.8368       5.0935          211
----------------------------------------------------------------------------------------------------
