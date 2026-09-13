# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Sep 13 02:52:26 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,391,447
ASIC Simulator                  10,000        10.49         1.05       1,591,759
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
Tag Lookup                           0.0180       0.0160       0.0375       0.0450       55,567
AND Search (3 tags)                  0.0363       0.0348       0.0492       0.0553       27,524
OR Search (3 tags)                   1.3982       1.3432       1.7436       1.9527          715
Merkle Verification                  5.3264       5.3040       5.3972       5.5820          188
Full Query Pipeline                  5.5156       5.4894       5.6308       6.0101          181
----------------------------------------------------------------------------------------------------
