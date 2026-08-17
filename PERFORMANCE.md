# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Aug 17 01:01:29 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.24         0.61       1,381,393
ASIC Simulator                  10,000        10.50         1.05       1,616,696
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
Tag Lookup                           0.0184       0.0165       0.0302       0.0481       54,342
AND Search (3 tags)                  0.0374       0.0356       0.0508       0.0615       26,765
OR Search (3 tags)                   1.6192       1.5734       2.0483       2.2613          618
Merkle Verification                  5.1772       5.1612       5.2565       5.4165          193
Full Query Pipeline                  5.3985       5.3899       5.5130       5.8012          185
----------------------------------------------------------------------------------------------------
