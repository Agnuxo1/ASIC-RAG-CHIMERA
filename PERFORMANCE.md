# ASIC-RAG-CHIMERA Performance Report
Generated: Mon Jun 29 03:29:51 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.22         0.62       1,385,681
ASIC Simulator                  10,000        10.57         1.06       1,620,355
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
Tag Lookup                           0.0184       0.0164       0.0350       0.0424       54,265
AND Search (3 tags)                  0.0360       0.0346       0.0470       0.0550       27,796
OR Search (3 tags)                   1.3703       1.3083       1.6878       1.8963          730
Merkle Verification                  5.1744       5.1699       5.2483       5.4356          193
Full Query Pipeline                  5.3225       5.3040       5.4039       5.5250          188
----------------------------------------------------------------------------------------------------
