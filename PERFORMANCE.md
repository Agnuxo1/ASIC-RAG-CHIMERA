# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Aug 16 01:04:10 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.25         0.45       1,904,157
ASIC Simulator                  10,000         8.27         0.83       2,150,631
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
Tag Lookup                           0.0162       0.0146       0.0329       0.0378       61,624
AND Search (3 tags)                  0.0315       0.0304       0.0416       0.0460       31,776
OR Search (3 tags)                   1.2499       1.1676       1.6584       2.0517          800
Merkle Verification                  3.5932       3.5845       3.6366       3.7397          278
Full Query Pipeline                  3.7075       3.6967       3.8053       3.8836          270
----------------------------------------------------------------------------------------------------
