# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Sep  4 02:40:44 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.90         0.59       1,449,502
ASIC Simulator                  10,000        10.47         1.05       1,623,986
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.12x
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
Tag Lookup                           0.0201       0.0180       0.0329       0.0508       49,852
AND Search (3 tags)                  0.0396       0.0385       0.0514       0.0606       25,242
OR Search (3 tags)                   1.6452       1.5917       2.0670       2.2838          608
Merkle Verification                  4.7875       4.7758       4.8674       4.9621          209
Full Query Pipeline                  4.9782       4.9725       5.1238       5.3257          201
----------------------------------------------------------------------------------------------------
