# LV06 Reservoir Computing Replication Guide

This guide describes how to replicate the successful physical reservoir computing experiment on the Lucky Miner LV06 (Bitmain BM1366 ASIC).

## 1. Hardware Calibration (The "Cool Reservoir" Strategy)

To achieve stable physical coupling without thermal shutdown, the hardware must be calibrated as follows:

- **Frequency**: 450 MHz
- **Core Voltage**: 1100 mV
- **Target Temperature**: < 40°C (Stable at ~34°C in our tests)

> [!IMPORTANT]
> Do not use default settings (550MHz) as they lead to thermal lockouts (75°C+) and decouple the sharing pipeline.

## 2. Software Setup

1. **Driver**: Use the included `universal_lv06_driver.py`. This SDK handles the HTTP configuration and the Stratum server for difficulty modulation.
2. **Environment**: Python 3.8+ (No external dependencies except `requests`).

## 3. Running the Experiment

Execute the definitive assessment script:

```bash
python lv06_reservoir_experiment_v4.py
```

### Key Parameters:
- `WINDOW_TIME = 2.0`: Increases the sampling window to allow for stable share arrival distributions.
- `D_BASE = 1.0`: High share density for frequency-to-rate encoding.

## 4. Expected Results

- **NRMSE (NARMA-10)**: 0.14 - 0.16
- **Shuffle Test**: Should yield > 0.20 NRMSE (proving temporal causality).
- **Shares**: ~500+ accepted shares in 45 minutes.

## 5. Directory Structure

- `/code`: The main experiment script.
- `/driver`: The hardware abstraction layer.
- `/results`: Summary of the validated run results.
