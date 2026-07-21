# ASIC-RAG-CHIMERA Performance Report
Generated: Tue Jul 21 02:14:18 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.29         0.62       1,372,357
ASIC Simulator                  10,000        10.64         1.06       1,604,237
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.17x
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
Tag Lookup                           0.0185       0.0165       0.0358       0.0407       53,929
AND Search (3 tags)                  0.0375       0.0360       0.0503       0.0590       26,700
OR Search (3 tags)                   1.5904       1.5220       1.9689       2.1825          629
Merkle Verification                  5.3223       5.3046       5.3981       5.6204          188
Full Query Pipeline                  5.4301       5.4159       5.5837       5.8231          184
----------------------------------------------------------------------------------------------------
