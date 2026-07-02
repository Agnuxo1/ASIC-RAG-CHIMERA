# ASIC-RAG-CHIMERA Performance Report
Generated: Thu Jul  2 02:52:06 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.20         0.62       1,389,596
ASIC Simulator                  10,000        10.64         1.06       1,612,930
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
Tag Lookup                           0.0183       0.0165       0.0277       0.0415       54,684
AND Search (3 tags)                  0.0362       0.0350       0.0478       0.0567       27,598
OR Search (3 tags)                   1.3199       1.2710       1.6075       1.7643          758
Merkle Verification                  5.2377       5.2222       5.3185       5.4254          191
Full Query Pipeline                  5.3523       5.3674       5.4598       5.5197          187
----------------------------------------------------------------------------------------------------
