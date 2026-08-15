# ASIC-RAG-CHIMERA Performance Report
Generated: Sat Aug 15 01:00:38 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.19         0.62       1,390,580
ASIC Simulator                  10,000        10.33         1.03       1,631,565
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
Tag Lookup                           0.0181       0.0164       0.0273       0.0427       55,337
AND Search (3 tags)                  0.0361       0.0348       0.0471       0.0573       27,726
OR Search (3 tags)                   1.5288       1.4822       1.8978       2.2624          654
Merkle Verification                  5.1315       5.1213       5.1907       5.3110          195
Full Query Pipeline                  5.3160       5.3117       5.4303       5.5324          188
----------------------------------------------------------------------------------------------------
