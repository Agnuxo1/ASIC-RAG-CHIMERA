# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Aug 23 01:05:13 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.96         0.60       1,437,193
ASIC Simulator                  10,000        10.36         1.04       1,648,012
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
Tag Lookup                           0.0203       0.0181       0.0443       0.0508       49,149
AND Search (3 tags)                  0.0400       0.0388       0.0514       0.0574       25,022
OR Search (3 tags)                   1.5593       1.5085       1.9628       2.2327          641
Merkle Verification                  4.7355       4.7173       4.8329       5.1409          211
Full Query Pipeline                  4.8413       4.8403       4.9586       5.0904          207
----------------------------------------------------------------------------------------------------
