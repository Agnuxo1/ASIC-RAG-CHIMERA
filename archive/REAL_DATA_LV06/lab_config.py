# lab_config.py
"""
Configuración Central del Laboratorio CHIMERA
Hardware: Lucky Miner LV06 (BM1387)
"""

# RED
PC_IP = "192.168.0.11"        # Tu PC (Donde corre el Bridge y los Experimentos)
MINER_IP = "192.168.0.15"     # El LV06
STRATUM_PORT = 3333           # Puerto de Minería
BRIDGE_API_PORT = 4029        # Puerto de Datos del Experimento

# HARDWARE SPECS
CHIPS_PER_LV06 = 1            # BM1387
CHIPS_PER_S9 = 189            # Antminer S9 estándar
LV06_RATED_GHS = 500.0        # Gigahashes nominales del LV06

# PARÁMETROS DE EXPERIMENTO
POLL_INTERVAL = 2.0           # Segundos (Time Anchor para proteger WiFi)
SAMPLE_SIZE = 5000            # Cuantos shares capturar para estadística robusta