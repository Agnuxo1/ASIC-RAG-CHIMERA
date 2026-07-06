# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jul  6 02:47:04 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.53         0.65       1,328,398
ASIC Simulator                  10,000        11.23         1.12       1,608,611
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.21x
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
Tag Lookup                           0.0181       0.0166       0.0233       0.0465       55,219
AND Search (3 tags)                  0.0374       0.0359       0.0497       0.0558       26,748
OR Search (3 tags)                   1.6563       1.5955       2.0314       2.3738          604
Merkle Verification                  5.2494       5.2194       5.3013       5.6377          190
Full Query Pipeline                  5.3626       5.3505       5.4655       5.5511          186
----------------------------------------------------------------------------------------------------
