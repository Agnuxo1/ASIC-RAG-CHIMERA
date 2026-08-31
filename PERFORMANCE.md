# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Aug 31 03:13:12 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.87         0.60       1,455,160
ASIC Simulator                  10,000        10.85         1.09       1,647,008
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.13x
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
Tag Lookup                           0.0198       0.0180       0.0384       0.0486       50,500
AND Search (3 tags)                  0.0406       0.0392       0.0533       0.0617       24,602
OR Search (3 tags)                   1.7185       1.6697       2.2123       2.3771          582
Merkle Verification                  4.6925       4.6615       4.7662       5.5627          213
Full Query Pipeline                  4.8903       4.8441       4.9690       6.4754          204
----------------------------------------------------------------------------------------------------
