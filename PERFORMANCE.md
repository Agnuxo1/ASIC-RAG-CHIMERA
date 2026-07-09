# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jul  9 02:34:38 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.25         0.62       1,379,662
ASIC Simulator                  10,000        10.92         1.09       1,590,787
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.15x
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
Tag Lookup                           0.0185       0.0166       0.0356       0.0417       53,922
AND Search (3 tags)                  0.0375       0.0360       0.0506       0.0604       26,649
OR Search (3 tags)                   1.6407       1.5822       2.0601       2.2413          610
Merkle Verification                  5.2562       5.2366       5.3372       5.6474          190
Full Query Pipeline                  5.4733       5.4556       5.6201       5.8410          183
----------------------------------------------------------------------------------------------------
