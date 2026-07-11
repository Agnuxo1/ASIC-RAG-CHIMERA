# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jul 11 02:13:47 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         5.70         0.49       1,755,612
ASIC Simulator                  10,000         8.00         0.80       2,139,583
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.22x
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
Tag Lookup                           0.0161       0.0145       0.0304       0.0409       62,151
AND Search (3 tags)                  0.0314       0.0304       0.0411       0.0471       31,858
OR Search (3 tags)                   1.2560       1.2127       1.5733       1.7654          796
Merkle Verification                  3.5764       3.5697       3.6248       3.6927          280
Full Query Pipeline                  3.7510       3.7187       3.8287       5.3705          267
----------------------------------------------------------------------------------------------------
