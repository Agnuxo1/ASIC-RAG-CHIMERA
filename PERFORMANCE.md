# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Jul 29 02:11:23 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.14         0.61       1,401,517
ASIC Simulator                  10,000        10.58         1.06       1,614,129
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
Tag Lookup                           0.0183       0.0165       0.0359       0.0412       54,590
AND Search (3 tags)                  0.0370       0.0357       0.0494       0.0571       27,010
OR Search (3 tags)                   1.6984       1.6152       2.2059       2.6315          589
Merkle Verification                  5.2496       5.2410       5.3361       5.4196          190
Full Query Pipeline                  5.3770       5.3675       5.5057       5.6123          186
----------------------------------------------------------------------------------------------------
