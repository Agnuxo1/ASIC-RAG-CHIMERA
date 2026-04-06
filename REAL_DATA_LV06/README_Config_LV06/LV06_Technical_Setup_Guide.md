# Lucky Miner LV06: Hardware Configuration & TNRC Setup Guide

This document provides the necessary instructions and technical specifications to configure the **Lucky Miner LV06 (BM1366 ASIC)** for **Thermodynamic Neuromorphic Reservoir Computing (TNRC)** experiments.

## 1. Hardware Overview
The Lucky Miner LV06 is a compact Bitcoin miner based on the Bitaxe (AxeOS) architecture. For our experiments, it serves as a physical reservoir where:
- **Input**: Data is injected via Stratum jobs (Merkle Root and nTime).
- **Physical State**: The ASIC's silicon reacts to the computational load.
- **Output (Entropy)**: High-resolution timing of solutions (jitter) and nonce distributions are harvested as the reservoir state.

## 2. Firmware Prerequisites
Ensure the device is running **AxeOS** (or compatible Bitaxe firmware). The experiments rely on the HTTP API provided by AxeOS for frequency and system control.

- **Default IP**: Usually assigned via DHCP (e.g., `192.168.0.15`).
- **Web UI**: Access via `http://[MINER_IP]`.

## 3. Frequency Tuning (The 525 MHz Regime)
The most critical adjustment for neuromorphic behavior is the frequency. Standard mining frequencies (approx. 450-500 MHz) or high-performance ones (600+ MHz) may not exhibit the optimal "Fading Memory" effect.

> [!IMPORTANT]
> A target frequency of **525 MHz** has been identified as the "sweet spot" where thermal and electrical dynamics provide maximum computational utility for non-linear tasks like NARMA-10.

### Automatic Configuration (Python)
Use a `PATCH` request to the `/api/system` endpoint followed by a reboot:
```python
import requests

def set_frequency(ip, freq_mhz):
    # AxeOS uses PATCH for system settings
    url = f"http://{ip}/api/system"
    payload = {"frequency": freq_mhz}
    response = requests.patch(url, json=payload, timeout=5)
    
    if response.status_code == 200:
        # Reboot is mandatory to apply the new PLL clock
        requests.post(f"http://{ip}/api/reboot")
        return True
    return False
```

## 4. Stratum Configuration (The Reservoir Bridge)
The LV06 must connect to a local **Stratum Server (Bridge)** that provides custom jobs for data injection.

### Required Settings in Miner Web UI:
- **Stratum URL**: `stratum+tcp://[YOUR_PC_IP]`
- **Port**: `3333` (Default for Chronos Bridge)
- **User**: `chimera.worker1`
- **Password**: `x`

### The "Chronos Bridge" Driver
The `chronos_bridge_v2.py` script acts as the middleware. It:
1. Handles the **mining.subscribe** and **mining.authorize** handshakes.
2. Maintains a **high-speed job flow**.
3. Captures the **Time Anchor** of every share with microsecond precision.

## 5. Crucial Adjustment: Ultra-Low Difficulty
Standard Bitcoin mining uses high difficulty, resulting in very few shares (outputs). For TNRC, we require high data density.

> [!WARNING]
> You MUST set a static difficulty of **0.001 or lower** (e.g., `0.0001`). This ensures the miner returns thousands of "shares" per minute, providing the high-frequency sampling required for reservoir computing.

**Python Implementation (Stratum Level):**
```python
def send_difficulty(conn, difficulty=0.0001):
    msg = {"id": None, "method": "mining.set_difficulty", "params": [difficulty]}
    conn.sendall((json.dumps(msg) + '\n').encode())
```

## 6. Data Injection (Input Coding)
To influence the reservoir state, data must be encoded into the Stratum jobs.
- **Merkle Root Injection**: Encode the input value `u` as a float converted to hex within `coinb1`.
- **nTime Perturbation**: Modulate the `ntime` parameter based on the input signal to further disturb the electrical state of the BM1366 chip.

## 7. Execution Workflow
To replicate the experiments:

1. **Kill existing miners**: Ensure no other Stratum servers are running on port 3333.
2. **Start the Bridge**: Run `python chronos_bridge_v2.py`.
3. **Configure Frequency**: Set miner to **525 MHz** and reboot.
4. **Launch Experiment**: Run the task script (e.g., `narma_reservoir_experiment_2.py`).
5. **Monitor Rate**: Ensure the execution reaches **>5 steps/second** to stay within the chip's physical memory window (Fading Memory).

## 8. State Harvesting
The reservoir state vector `X(t)` is typically formed by:
1. **Deltas (Jitter)**: `log1p(t[i] - t[i-1])` between shares.
2. **Nonce Entropy**: `int(nonce[-4:], 16) / 0xFFFF` (normalizing the tail of the nonces).
3. **Statistical Moments**: Mean and standard deviation of share arrival times within the sampling window.

---
*Documented by the ASIC-RAG-CHIMERA Research Team.*
