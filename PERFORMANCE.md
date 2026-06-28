# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jun 28 03:01:51 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,387,912
ASIC Simulator                  10,000        11.83         1.18       1,578,447
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
Tag Lookup                           0.0183       0.0165       0.0340       0.0417       54,652
AND Search (3 tags)                  0.0374       0.0359       0.0498       0.0587       26,761
OR Search (3 tags)                   1.5168       1.4439       1.9278       2.1735          659
Merkle Verification                  5.2271       5.2100       5.3142       5.4406          191
Full Query Pipeline                  5.4443       5.4113       5.5566       5.9404          184
----------------------------------------------------------------------------------------------------
