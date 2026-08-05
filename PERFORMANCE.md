# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Aug  5 02:05:52 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.75         0.58       1,481,601
ASIC Simulator                  10,000        10.38         1.04       1,667,528
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
Tag Lookup                           0.0208       0.0187       0.0333       0.0500       48,192
AND Search (3 tags)                  0.0399       0.0386       0.0511       0.0584       25,055
OR Search (3 tags)                   1.4704       1.4091       1.8411       2.1677          680
Merkle Verification                  4.6374       4.6066       4.7249       5.4286          216
Full Query Pipeline                  4.7889       4.7688       4.9055       5.6689          209
----------------------------------------------------------------------------------------------------
