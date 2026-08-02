# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Aug  2 02:24:14 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.35         0.63       1,361,466
ASIC Simulator                  10,000        10.89         1.09       1,572,889
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.16x
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
Tag Lookup                           0.0184       0.0165       0.0331       0.0428       54,327
AND Search (3 tags)                  0.0371       0.0357       0.0490       0.0611       26,950
OR Search (3 tags)                   1.5462       1.4947       1.9198       2.1400          647
Merkle Verification                  5.3173       5.3104       5.4131       5.5157          188
Full Query Pipeline                  5.4529       5.4688       5.5812       5.6841          183
----------------------------------------------------------------------------------------------------
