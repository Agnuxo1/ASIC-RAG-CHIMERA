Here is the complete, professional technical documentation in English. You can save this file directly as `README_Config_LV06.md` in your `D:\ASIC_RAG\REAL_DATA_LV06\` folder.

It covers everything from the firmware settings to the Python bridge configuration, ensuring any computer scientist can replicate your NRMSE < 0.20 results.

***

# ASIC-RAG-CHIMERA: Lucky Miner LV06 (BM1366) Replication Guide

**Project:** Physical Reservoir Computing using 5nm Bitcoin ASICs
**Hardware Target:** Lucky Miner LV06 (Bitaxe / BM1366 Chip)
**Author:** ASIC-RAG-CHIMERA Team
**Date:** December 2025

---

## 1. Overview
This document outlines the exact hardware and software configuration required to replicate the **Physical Reservoir Computing** experiments. The goal is to transition the ASIC from a standard SHA-256 hasher into a non-linear chaotic oscillator capable of solving time-series tasks (NARMA-10).

**Critical Requirement:** To achieve high-frequency sampling (>20 Hz) and capture the "Fading Memory" property, the miner must be configured to submit shares continuously at extremely low difficulty, rather than searching for valid Bitcoin blocks.

---

## 2. Hardware Prerequisites
*   **Device:** Lucky Miner LV06 (or any Bitaxe Ultra variant based on BM1366).
*   **Power Supply:** 5V / 4A DC Adapter (Stable voltage is crucial for signal consistency).
*   **Cooling:** Standard fan (must be capable of manual RPM control).
*   **Host Machine:** PC/Laptop running Python 3.9+ (Windows/Linux).
*   **Network:** Low-latency Local LAN (Ethernet recommended for Host, 2.4GHz Wi-Fi for ASIC).

---

## 3. Firmware Configuration (AxeOS)
The LV06 runs **AxeOS**. You must access the web interface (usually `http://192.168.x.x`) to configure the physical parameters.

### A. Stratum Settings (The Connection)
Navigate to **Settings > Stratum**:
*   **URL:** `stratum+tcp://<YOUR_HOST_IP>:3333`
    *   *Example:* `stratum+tcp://192.168.0.11:3333`
*   **User:** `chimera`
*   **Password:** `x`
*   **Note:** The miner must point to your Python script, NOT a real mining pool.

### B. Physical Tuning (The "Edge of Chaos")
Navigate to **Settings > System / Power**:
This is the most critical step. We need the chip to be reactive but stable.

*   **Frequency:** **525 MHz**
    *   *Scientific Context:* This is the phase transition point where 1/f noise (Pink Noise) maximizes. Below 400 MHz is too linear; above 600 MHz becomes saturated white noise.
*   **Core Voltage:** **990 mV** (Default) or **970 mV**
    *   *Tuning:* If the miner crashes at 525 MHz, increase to 990 mV. If it is too stable (perfect linearity), lower slightly to induce thermal noise.
*   **Fan Speed:** **Manual Mode @ ~5000 RPM**
    *   *Reason:* We need "Thermal Memory". If the fan is on Auto and cools the chip instantly to 30°C, the memory effect vanishes. Maintaining the chip around **50°C - 60°C** improves the reservoir quality.

---

## 4. Software Environment (The Driver)
The experiments rely on a custom Python bridge that translates Stratum protocol into Reservoir Computing inputs.

### Requirements
```bash
pip install numpy scikit-learn requests
```

### The "Chronos Bridge" Driver
The script `narma_reservoir_experiment_2.py` (or `chronos_bridge_v2.py`) acts as a local Stratum Pool.
**Key Driver Logic (How it works):**
1.  **Injection:** It takes a floating point value (e.g., `0.354`) from the NARMA task.
2.  **Encoding:** It embeds this value into the **Coinbase Script** or manipulates the `nTime` field of the block template.
3.  **Broadcasting:** It sends this "perturbed" block to the LV06.
4.  **Sampling:** It listens for "Shares" (partial hash solutions) returning from the ASIC.

---

## 5. Critical Tuning for Replication
If you are getting `0 shares` or `0.1 steps/s`, you must apply these specific settings in the Python script constants.

### A. Difficulty Floor (The "Machine Gun" Mode)
The standard Stratum difficulty is `1` (too hard). For Reservoir Computing, we need a continuous stream of data.
*   **Setting:** `DIFFICULTY = 0.001` (or `0.0001`)
*   **Explanation:** This lowers the target threshold so the ASIC generates valid shares thousands of times per second. This allows us to measure **Density** and **Jitter** instantly.

### B. Time-Window Sampling
We do not wait for specific events. We sample the silicon state at a fixed frequency (Hz).
*   **Window Size:** `0.04s` to `0.1s` (25 Hz to 10 Hz).
*   **Logic:** The script counts how many shares arrive in exactly 0.04 seconds.
    *   *Result:* 0 shares = Low activation.
    *   *Result:* 50 shares = High activation.
    *   *Result:* Jitter variation = Non-linear transformation.

---

## 6. Step-by-Step Execution Protocol

1.  **Start the Bridge:**
    Run the Python script on your Host PC **first**.
    ```powershell
    python narma_reservoir_experiment_2.py
    ```
    *Output:* `[SERVER] Listening on 0.0.0.0:3333`

2.  **Boot the Miner:**
    Power on the LV06. Ensure it is connected to the same Wi-Fi/LAN.
    *Watch the OLED Screen:* It should show "Stratum Connected" and the hash rate should spike.

3.  **Verify Handshake:**
    The Python terminal should show:
    ```text
    [STRATUM] Subscribe from...
    [STRATUM] Authorize...
    [STRATUM] Setting Difficulty to 0.001
    ```

4.  **Observation:**
    You should see a stream of data:
    ```text
    Step 10/300 | Input: 0.45 | Shares: 12 | Rate: 25.0 steps/s
    Step 11/300 | Input: 0.22 | Shares: 8  | Rate: 25.1 steps/s
    ```

5.  **Result Analysis:**
    Upon completion, the script calculates the **NRMSE**.
    *   **Target:** NRMSE < 0.20 (Valid Neuromorphic Behavior).

---

## 7. Troubleshooting

**Problem: Miner connects but sends 0 shares.**
*   **Fix:** The Difficulty is too high. Change `DIFFICULTY` in the Python script to `0.001` or `0.0001`.
*   **Fix:** Ensure `coinbase` injection format is correct (Standard Bitcoin Header structure).

**Problem: "Socket Error" or "Connection Refused".**
*   **Fix:** Check Windows Firewall. Allow Port `3333` for Python.
*   **Fix:** Verify the Host IP hasn't changed (use `ipconfig`).

**Problem: High NRMSE (>0.4) / No Learning.**
*   **Fix:** The chip is too stable.
    1.  Lower Fan speed to increase temp to 60°C.
    2.  Lower Voltage to 970mV to introduce electrical noise.
    3.  Ensure Sampling Rate is fast (>10 Hz) to catch the thermal memory effect.

---

**Disclaimer:** This configuration runs the ASIC in a non-standard loop. While safe for short durations, continuous operation at high temperatures (>75°C) may degrade the hardware. Run experiments in supervised batches.