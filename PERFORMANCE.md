# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Aug 14 01:30:32 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.21         0.45       1,919,739
ASIC Simulator                  10,000         7.84         0.78       2,168,544
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
Tag Lookup                           0.0157       0.0144       0.0225       0.0402       63,589
AND Search (3 tags)                  0.0313       0.0304       0.0404       0.0480       31,959
OR Search (3 tags)                   1.1774       1.1243       1.4899       1.7607          849
Merkle Verification                  3.9318       3.9254       3.9785       4.0295          254
Full Query Pipeline                  4.0617       4.0474       4.1487       4.4002          246
----------------------------------------------------------------------------------------------------
