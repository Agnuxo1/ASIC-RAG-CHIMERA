# LV06 Quickstart Guide

Get up and running with the LV06 Reservoir Computing SDK in under 5 minutes.

## 1. Prerequisites
- Python 3.8+
- Lucky Miner LV06 on the same network
- `requests` and `numpy` installed: `pip install requests numpy`

## 2. Basic Setup (The "Hello World" of Reservoir Computing)
Save the following as `quickstart.py` in the same directory as `universal_lv06_driver.py`.

```python
from universal_lv06_driver import LV06Config, LV06StratumServer
import time

# 1. Hardware Config
MINER_IP = "192.168.0.15"  # CHANGE THIS TO YOUR MINER IP
cfg = LV06Config(MINER_IP)
cfg.set_frequency(500)     # Set to 500MHz

# 2. Start Stratum Server
server = LV06StratumServer(host="0.0.0.0", port=3333)
server.start()

print("Waiting for miner to connect...")

while not server.connection_active:
    time.sleep(1)

# 3. Rate-Encoded Injection Loop
D_BASE = 25.0
try:
    for i in range(100):
        val = (i % 10) / 10.0  # Simple oscillating input
        server.inject_rate(val, d_base=D_BASE) # PHYSICS ACTIVE HERE
        
        time.sleep(1.0)  # 1Hz sampling
        
        shares = server.harvest_state()
        print(f"Step {i}: Injected {val} | Target Difficulty: {D_BASE/(val+0.01):.1f} | Shares={len(shares)}")
        
finally:
    server.stop()
```

## 3. Configuring the LV06 Hardware
To connect the miner to your SDK:
1.  Open the Lucky Miner web interface.
2.  Navigate to **Miner Configuration**.
3.  Set **Pool 1** to your computer's IP address (e.g., `stratum+tcp://192.168.0.10:3333`).
4.  Set **User** to `worker1` and **Password** to `x`.
5.  Click **Save & Apply**.

## 4. Next Steps
- Implement **Jitter Extraction**: Look at the `time` field in the shares returned by `harvest_state()`.
- Run the **NARMA-10 Benchmark**: Use the SDK to collect data for complex non-linear time series estimation.

---
*Created by the ASIC-RAG-CHIMERA Research Team.*
