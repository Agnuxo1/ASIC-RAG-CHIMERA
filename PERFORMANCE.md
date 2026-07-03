# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jul  3 02:35:41 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.29         0.63       1,371,118
ASIC Simulator                  10,000        10.65         1.06       1,609,860
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
Tag Lookup                           0.0185       0.0165       0.0382       0.0417       54,089
AND Search (3 tags)                  0.0370       0.0355       0.0493       0.0584       27,035
OR Search (3 tags)                   1.6623       1.5988       2.0712       2.2653          602
Merkle Verification                  5.2242       5.2132       5.2881       5.3711          191
Full Query Pipeline                  5.3570       5.3312       5.4559       6.0650          187
----------------------------------------------------------------------------------------------------
