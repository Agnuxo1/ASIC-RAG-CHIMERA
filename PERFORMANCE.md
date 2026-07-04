# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jul  4 02:32:32 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         4.13         0.36       2,423,789
ASIC Simulator                  10,000         6.39         0.64       2,826,748
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
Tag Lookup                           0.0146       0.0135       0.0261       0.0369       68,402
AND Search (3 tags)                  0.0273       0.0264       0.0361       0.0404       36,603
OR Search (3 tags)                   0.8470       0.8181       1.0595       1.2951        1,181
Merkle Verification                  2.6944       2.6724       2.7953       3.0787          371
Full Query Pipeline                  2.7940       2.7823       2.8840       2.9595          358
----------------------------------------------------------------------------------------------------
