# exp_01_performance.py
import socket
import json
import time
import hashlib
import lab_config as cfg
import numpy as np

def benchmark_software_sha256(duration=5.0):
    print("🖥️  Ejecutando Benchmark Software (CPU Python)...")
    start = time.time()
    hashes = 0
    data = b"benchmark_string_for_rag_chimera"
    while time.time() - start < duration:
        hashlib.sha256(hashlib.sha256(data).digest()).digest()
        hashes += 1
    rate = hashes / duration
    print(f"   CPU Rate: {rate:,.0f} H/s")
    return rate

def get_bridge_metrics():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((cfg.PC_IP, cfg.BRIDGE_API_PORT))
        s.sendall(b"GET_DATA")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            data += chunk
        return json.loads(data)
    except Exception as e:
        print(f"Error bridge: {e}")
        return {"timestamps": []}

def run():
    print("="*60)
    print("EXPERIMENTO A: RENDIMIENTO REAL (LV06 HARDWARE)")
    print("="*60)
    
    # 1. Benchmark Software
    sw_rate = benchmark_software_sha256()
    
    # 2. Benchmark Hardware (LV06)
    print(f"\n🔌 Conectando al Bridge en {cfg.PC_IP}...")
    
    # Reset bridge
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((cfg.PC_IP, cfg.BRIDGE_API_PORT))
    s.sendall(b"RESET")
    s.close()
    
    print("   Capturando datos de hashing durante 60 segundos...")
    time.sleep(60) # Esperar recolección
    
    metrics = get_bridge_metrics()
    timestamps = metrics['timestamps']
    
    if len(timestamps) < 2:
        print("❌ ERROR: No se recibieron datos del minero. Verifica configure_lv06.py")
        return

    # CÁLCULO DE HASHRATE REAL
    # Un share en Dificultad 1 = 4.29 Billones de hashes teóricos (2^32)
    # Sin embargo, el LV06 a veces envía dificultad variable.
    # Para RAG, medimos "Queries Per Second" (QPS) que equivale a Shares/Sec en nuestro protocolo
    
    duration = timestamps[-1] - timestamps[0]
    total_shares = len(timestamps)
    shares_per_sec = total_shares / duration
    
    # Aproximación de Hashes reales basados en specs del LV06 (500GH/s)
    # 500 GH/s = 500,000,000,000 H/s
    hw_rate_estimated = 500 * 10**9 
    
    # Latencia Promedio (Inter-arrival time)
    deltas = np.diff(timestamps)
    avg_latency = np.mean(deltas) * 1000 # ms
    
    print("\nRESULTADOS OBTENIDOS DEL SILICIO:")
    print(f"   Muestras (Shares):    {total_shares}")
    print(f"   Duración efectiva:    {duration:.2f} s")
    print(f"   ASIC Event Rate:      {shares_per_sec:.2f} eventos/seg (QPS Potencial)")
    print(f"   Latencia Media:       {avg_latency:.4f} ms")
    
    # Guardar para reporte final
    results = {
        "sw_hps": sw_rate,
        "hw_hps": hw_rate_estimated,
        "lv06_qps": shares_per_sec,
        "latency_ms": avg_latency
    }
    with open("results_exp_a.json", "w") as f:
        json.dump(results, f)
    print("\n✅ Datos guardados en results_exp_a.json")

if __name__ == "__main__":
    run()