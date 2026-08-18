# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Aug 18 00:59:26 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.48         0.64       1,337,726
ASIC Simulator                  10,000        10.44         1.04       1,594,741
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.19x
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
Tag Lookup                           0.0186       0.0166       0.0327       0.0468       53,673
AND Search (3 tags)                  0.0385       0.0368       0.0528       0.0616       25,974
OR Search (3 tags)                   1.7870       1.6988       2.2507       2.6453          560
Merkle Verification                  5.1371       5.1220       5.2209       5.5380          195
Full Query Pipeline                  5.3280       5.3197       5.4703       5.6552          188
----------------------------------------------------------------------------------------------------
