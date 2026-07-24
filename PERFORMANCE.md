# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jul 24 02:15:59 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.31         0.46       1,884,908
ASIC Simulator                  10,000         8.04         0.80       2,157,428
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
Tag Lookup                           0.0162       0.0145       0.0351       0.0402       61,675
AND Search (3 tags)                  0.0318       0.0307       0.0428       0.0470       31,416
OR Search (3 tags)                   1.2789       1.2427       1.6157       1.8217          782
Merkle Verification                  3.5940       3.5878       3.6376       3.6774          278
Full Query Pipeline                  3.7310       3.7251       3.8181       3.8873          268
----------------------------------------------------------------------------------------------------
