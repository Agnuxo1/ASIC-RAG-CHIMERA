# exp_02_reservoir_entropy.py
import socket
import json
import time
import numpy as np
import lab_config as cfg
import matplotlib.pyplot as plt

def calculate_entropy(data):
    """Calcula entropía de Shannon de los intervalos de tiempo"""
    hist, _ = np.histogram(data, bins=20, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def run():
    print("="*60)
    print("EXPERIMENTO B: CARACTERIZACIÓN DEL RESERVORIO FÍSICO")
    print("="*60)
    
    print("📡 Capturando micro-variaciones temporales (Jitter)...")
    
    # Reset y Captura
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((cfg.PC_IP, cfg.BRIDGE_API_PORT))
    s.sendall(b"RESET")
    s.close()
    
    wait_time = 300
    print(f"   Muestreando flujo de caos durante {wait_time}s...")
    for i in range(wait_time):
        print(f"\r   Progreso: {i+1}/{wait_time}s", end="")
        time.sleep(1)
    print()
    
    # Obtener datos
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((cfg.PC_IP, cfg.BRIDGE_API_PORT))
    s.sendall(b"GET_DATA")
    raw = b""
    while True:
        c = s.recv(4096)
        if not c: break
        raw += c
    data = json.loads(raw)
    
    timestamps = np.array(data['timestamps'])
    if len(timestamps) < 10:
        print("❌ Datos insuficientes.")
        return

    # Análisis de Dinámica
    deltas = np.diff(timestamps) * 1000 # ms
    
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    cv = std_delta / mean_delta # Coeficiente de Variación
    entropy = calculate_entropy(deltas)
    
    print("\nMETRICAS DE RESERVORIO (Datos Reales):")
    print(f"   Estabilidad Temporal (Mean): {mean_delta:.4f} ms")
    print(f"   Jitter Físico (Std Dev):     {std_delta:.4f} ms")
    print(f"   Coeficiente de Variación:    {cv:.4f} (CV > 0 indica caos explotable)")
    print(f"   Entropía de Shannon:         {entropy:.4f} bits/símbolo")
    
    validation = "EXITOSA" if cv > 0.05 else "BAJA ENTROPÍA"
    print(f"   Validación de Caos:          {validation}")

    # Guardar
    results = {
        "entropy": entropy,
        "jitter_ms": std_delta,
        "cv": cv,
        "sample_count": len(deltas)
    }
    with open("results_exp_b.json", "w") as f:
        json.dump(results, f)
    print("\n✅ Datos guardados en results_exp_b.json")

if __name__ == "__main__":
    run()