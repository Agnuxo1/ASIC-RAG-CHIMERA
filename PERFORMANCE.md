# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Jun 27 02:46:35 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.39         0.64       1,352,834
ASIC Simulator                  10,000        10.80         1.08       1,554,945
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.15x
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
Tag Lookup                           0.0186       0.0166       0.0328       0.0445       53,679
AND Search (3 tags)                  0.0370       0.0355       0.0494       0.0579       26,995
OR Search (3 tags)                   1.4344       1.3772       1.8017       1.9604          697
Merkle Verification                  5.3210       5.3059       5.3988       5.5201          188
Full Query Pipeline                  5.4336       5.4198       5.5194       5.7408          184
----------------------------------------------------------------------------------------------------
