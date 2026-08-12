# ASIC-RAG-CHIMERA Performance Report
Generated: Wed Aug 12 01:29:21 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,390,126
ASIC Simulator                  10,000        10.77         1.08       1,586,091
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
Tag Lookup                           0.0184       0.0165       0.0363       0.0418       54,321
AND Search (3 tags)                  0.0373       0.0360       0.0502       0.0599       26,789
OR Search (3 tags)                   1.5943       1.5584       1.9978       2.2524          627
Merkle Verification                  5.2613       5.2496       5.3385       5.4186          190
Full Query Pipeline                  5.4365       5.4000       5.5788       6.7598          184
----------------------------------------------------------------------------------------------------
