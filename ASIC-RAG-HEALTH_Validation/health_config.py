"""
ASIC-RAG-HEALTH: Configuration File
Phase 2: Physical Hardware Validation
"""

# NETWORK CONFIGURATION
# ---------------------
PC_IP = "192.168.0.11"        # The Researcher's PC (Server)
MINER_IP = "192.168.0.15"     # The Lucky Miner LV06
STRATUM_PORT = 3333           # Mining Protocol Port
API_PORT = 4029               # Bridge Data API

# HARDWARE PROFILES (For Comparative Analysis)
# ---------------------
HARDWARE_SPECS = {
    "LV06_REAL": {
        "name": "Lucky Miner LV06 (Measured)",
        "chips": 1,
        "power_watts": 9.0,      # Ultra-low power mode
        "cost_usd": 40.0,
        "type": "ASIC (Rural Node)"
    },
    "S9_EXTRAPOLATED": {
        "name": "Antminer S9 (Projected)",
        "chips": 189,            # S9 has 189 BM1387 chips
        "power_watts": 1323.0,
        "cost_usd": 250.0,
        "type": "ASIC (City Hub)"
    },
    "RTX3090_REF": {
        "name": "NVIDIA RTX 3090 (Reference)",
        "chips": 1,
        "power_watts": 350.0,
        "cost_usd": 1500.0,
        "hashrate_ghs": 0.120,   # ~120 MH/s on SHA256d (GPUs are bad at this)
        "type": "GPU (Traditional)"
    }
}

# CLINIC SIMULATION PARAMETERS (Calibrated to Seid-Mehammed et al. 2024 protocol)
# ---------------------
DAILY_PATIENTS = 50              # Avg patients per rural clinic/day
RECORDS_PER_PATIENT = 5          # Vitals, Diagnosis, Script, Notes, Lab
BLOCK_DIFFICULTY = 0.08          # Calibrated Difficulty: Target block time ~0.83s