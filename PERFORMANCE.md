# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jul 18 02:04:37 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.27         0.62       1,375,971
ASIC Simulator                  10,000        10.36         1.04       1,623,620
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.18x
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
Tag Lookup                           0.0190       0.0165       0.0388       0.0475       52,710
AND Search (3 tags)                  0.0360       0.0347       0.0482       0.0553       27,754
OR Search (3 tags)                   1.3999       1.3435       1.7337       1.9649          714
Merkle Verification                  5.2582       5.2586       5.3458       5.4364          190
Full Query Pipeline                  5.3789       5.3675       5.5375       5.8642          186
----------------------------------------------------------------------------------------------------
