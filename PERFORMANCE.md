# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jul 23 02:24:16 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,391,648
ASIC Simulator                  10,000        10.30         1.03       1,616,284
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.16x
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
Tag Lookup                           0.0185       0.0165       0.0387       0.0412       54,153
AND Search (3 tags)                  0.0368       0.0354       0.0489       0.0577       27,163
OR Search (3 tags)                   1.4670       1.3960       1.8305       2.2258          682
Merkle Verification                  5.1595       5.1486       5.2235       5.2962          194
Full Query Pipeline                  5.2735       5.2716       5.3644       5.4962          190
----------------------------------------------------------------------------------------------------
