# NARMA-10 Dataset: Hardware Reservoir Computing (Bitmain BM1366)

## 1. Description
This dataset contains high-fidelity telemetry from a physical **Thermodynamic Neuromorphic Reservoir Computer (TNRC)** based on the Bitmain BM1366 ASIC (Lucky Miner LV06). 

The data captures the relationship between a complex input sequence (**NARMA-10**) and the resulting physical entropy (jitter and nonces) harvested from the silicon substrate operating at **525 MHz**.

## 2. Benchmark Summary
- **Benchmark Name:** NARMA-10
- **System:** Physical BM1366 ASIC (Lucky Miner LV06)
- **Clock Frequency:** 525 MHz
- **Injection Rate:** 25 Hz (40ms per step)
- **Sample Size:** 2000 steps
- **NRMSE (Performance):** **0.170225**
- **Sparsity:** 0% (2000/2000 successful steps)

## 3. Dataset Schema (`narma10_telemetry.csv`)
- `step`: Time step index (0-1999).
- `u_input`: The NARMA-10 input value injected into the Stratum Merkle Root.
- `y_target`: The theoretical NARMA-10 target value.
- `jitter_n`: The inter-arrival time jitter (logged) between mining solutions (shares).
- `nonce_n`: Normalized nonce entropy harvested from the hardware.

## 4. Usage for Researchers
This data is intended for researchers in **Reservoir Computing** and **Thermodynamic Computing**. It proves that Bitcoin mining ASICs possess intrinsic **Fading Memory** and non-linear transformation capabilities required for complex time-series prediction.

To verify the "Official" status of this benchmark, the readout layer (Ridge Regression) should be trained on the first 80% of steps and tested on the remaining 20%.

---
**Official Submission:** ASIC-RAG-CHIMERA Research Team (Agnuxo1/ASIC-RAG-CHIMERA)
**Platform:** Kaggle / GitHub Open Hardware Project
