# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Aug 24 01:02:39 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.42         0.64       1,346,896
ASIC Simulator                  10,000        10.91         1.09       1,606,605
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.19x
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
Tag Lookup                           0.0182       0.0162       0.0295       0.0459       55,049
AND Search (3 tags)                  0.0391       0.0372       0.0540       0.0636       25,563
OR Search (3 tags)                   1.8933       1.8230       2.4185       2.7275          528
Merkle Verification                  5.3531       5.3223       5.4351       6.4855          187
Full Query Pipeline                  5.5374       5.4951       5.6838       7.2407          181
----------------------------------------------------------------------------------------------------
