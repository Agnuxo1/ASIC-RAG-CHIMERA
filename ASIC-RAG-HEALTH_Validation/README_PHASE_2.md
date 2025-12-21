# Phase 2: LV06 "Clinic Node" Validation

This suite validates the **Lucky Miner LV06** as a replacement for GPUs in rural healthcare blockchains.

## 1. Hardware Setup
*   **Miner:** LV06 connected to WiFi (`192.168.0.15`).
*   **PC:** Connected to same network (`192.168.0.11`).
*   **Power:** LV06 plugged in (Blue LED breathing).

## 2. Configuration (Crucial)
You must point the miner to this PC.
Run the config tool from Phase 1 if you haven't:
```bash
python configure_lv06.py --miner-ip 192.168.0.15 --pc-ip 192.168.0.11