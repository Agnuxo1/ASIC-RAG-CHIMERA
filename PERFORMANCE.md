# ASIC-RAG-CHIMERA Performance Report
Generated: Sun Jul  5 02:41:53 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.29         0.63       1,370,898
ASIC Simulator                  10,000        10.95         1.09       1,619,267
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
Tag Lookup                           0.0182       0.0165       0.0311       0.0424       54,863
AND Search (3 tags)                  0.0395       0.0377       0.0535       0.0598       25,346
OR Search (3 tags)                   1.5517       1.5057       1.9015       2.1254          644
Merkle Verification                  5.2854       5.2714       5.3667       5.4658          189
Full Query Pipeline                  5.5207       5.5154       5.6593       5.7451          181
----------------------------------------------------------------------------------------------------
