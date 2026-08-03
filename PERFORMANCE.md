# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Aug  3 02:27:03 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.88         0.60       1,453,217
ASIC Simulator                  10,000         9.65         0.96       1,709,796
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.18x
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
Tag Lookup                           0.0215       0.0186       0.0471       0.0509       46,596
AND Search (3 tags)                  0.0409       0.0397       0.0515       0.0567       24,443
OR Search (3 tags)                   1.2738       1.2255       1.5242       1.8174          785
Merkle Verification                  4.8114       4.7872       4.9555       5.0202          208
Full Query Pipeline                  4.9577       4.9484       5.0837       5.1673          202
----------------------------------------------------------------------------------------------------
