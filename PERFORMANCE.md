# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jun 26 02:56:53 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.25         0.45       1,904,191
ASIC Simulator                  10,000         8.50         0.85       2,148,312
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
Tag Lookup                           0.0163       0.0146       0.0276       0.0406       61,412
AND Search (3 tags)                  0.0317       0.0307       0.0416       0.0485       31,501
OR Search (3 tags)                   1.3002       1.2808       1.7165       2.0743          769
Merkle Verification                  3.7125       3.7089       3.7595       3.8215          269
Full Query Pipeline                  3.8668       3.8621       3.9917       4.2049          259
----------------------------------------------------------------------------------------------------
