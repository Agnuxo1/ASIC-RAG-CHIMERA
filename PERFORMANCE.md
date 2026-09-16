# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Sep 16 03:04:57 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         6.05         0.52       1,652,472
ASIC Simulator                  10,000         8.25         0.83       1,976,885
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.20x
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
Tag Lookup                           0.0188       0.0164       0.0357       0.0469       53,298
AND Search (3 tags)                  0.0335       0.0327       0.0422       0.0500       29,847
OR Search (3 tags)                   1.0873       1.0538       1.2968       1.3377          920
Merkle Verification                  4.0710       4.0706       4.1368       4.1873          246
Full Query Pipeline                  4.1800       4.1803       4.2612       4.3140          239
----------------------------------------------------------------------------------------------------
