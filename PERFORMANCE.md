# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jul 26 02:25:45 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.21         0.62       1,387,924
ASIC Simulator                  10,000        10.93         1.09       1,588,809
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
Tag Lookup                           0.0187       0.0165       0.0375       0.0422       53,410
AND Search (3 tags)                  0.0360       0.0350       0.0468       0.0550       27,767
OR Search (3 tags)                   1.4280       1.3354       1.9116       2.2643          700
Merkle Verification                  5.2162       5.2046       5.2903       5.3732          192
Full Query Pipeline                  5.3547       5.3548       5.4551       5.5062          187
----------------------------------------------------------------------------------------------------
