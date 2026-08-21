# ASIC-RAG-CHIMERA Performance Report
Generated: Fri Aug 21 01:03:52 UTC 2026

Running hashlib benchmark...
Running ASIC simulator benchmark...
Running CHIMERA GPU benchmark...
CHIMERA integration not available

================================================================================
HASH BENCHMARK RESULTS
================================================================================
Implementation              Iterations   Total (ms)    Mean (µs)           H/sec
--------------------------------------------------------------------------------
hashlib (Python)                10,000         7.23         0.62       1,382,747
ASIC Simulator                  10,000        10.37         1.04       1,609,192
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
Tag Lookup                           0.0181       0.0160       0.0382       0.0420       55,186
AND Search (3 tags)                  0.0412       0.0376       0.0616       0.0737       24,271
OR Search (3 tags)                   1.4323       1.3740       1.7689       2.1826          698
Merkle Verification                  5.2126       5.2067       5.2755       5.3367          192
Full Query Pipeline                  5.3189       5.3151       5.4142       5.5081          188
----------------------------------------------------------------------------------------------------
