# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jun 21 03:42:32 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.23         0.62       1,382,171
ASIC Simulator                  10,000        12.56         1.26       1,517,168
--------------------------------------------------------------------------------

Speedup vs baseline (hashlib):
  ASIC Simulator: 1.10x
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
Tag Lookup                           0.0178       0.0160       0.0268       0.0451       56,090
AND Search (3 tags)                  0.0396       0.0381       0.0529       0.0597       25,277
OR Search (3 tags)                   1.6225       1.5506       2.0498       2.4629          616
Merkle Verification                  5.2376       5.2227       5.3106       5.4042          191
Full Query Pipeline                  5.3961       5.3950       5.5075       5.6094          185
----------------------------------------------------------------------------------------------------
