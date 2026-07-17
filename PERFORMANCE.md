# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Jul 17 02:14:49 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.42         0.64       1,347,429
ASIC Simulator                  10,000        10.39         1.04       1,627,970
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.21x
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
Tag Lookup                           0.0183       0.0165       0.0295       0.0464       54,552
AND Search (3 tags)                  0.0351       0.0339       0.0459       0.0531       28,464
OR Search (3 tags)                   1.3120       1.2642       1.5980       1.6557          762
Merkle Verification                  5.2177       5.2034       5.2932       5.3603          192
Full Query Pipeline                  5.3459       5.3556       5.4409       5.4879          187
----------------------------------------------------------------------------------------------------
