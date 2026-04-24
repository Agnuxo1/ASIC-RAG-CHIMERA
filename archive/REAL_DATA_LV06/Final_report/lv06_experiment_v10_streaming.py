#!/usr/bin/env python3
"""
================================================================================
LV06 EXPERIMENT V10: ASYNCHRONOUS STREAMING PROTOCOL
"The Flow State"
================================================================================
Solución a los problemas de V7/V9:
1. Desacoplamiento total de Inyección (Input) y Recolección (Output).
2. Uso de Job IDs para correlacionar Input->Output a pesar de la latencia de red.
3. Mantenimiento del estado térmico del ASIC mediante flujo continuo (sin pausas).

Este script trata al ASIC como un sistema de flujo continuo, no como una calculadora
de pasos discretos.

Author: Fran / Agnuxo + Claude Architecture
Date: December 2025
================================================================================
"""

import socket
import threading
import json
import time
import struct
import queue
import random
import math
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import requests

# Thread-safe logging
PRINT_LOCK = threading.Lock()
def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)

# =============================================================================
# CONFIGURACIÓN OPTIMIZADA (V10)
# =============================================================================
class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    
    # Streaming Settings
    STREAM_RATE_HZ = 0.5        # 1 input cada 2 segundos (estabilidad ASIC)
    
    # Difficulty Settings (Rate Encoding)
    D_BASE = 0.005              # Mucho más bajo para inundar de shares
    D_MODULATION = 0.005
    
    # Experiment
    NARMA_STEPS = 500           # Escalado de 100 a 500
    WARMUP_SEC = 10             # Calentamiento extendido
    MINER_IP = "192.168.0.15"              # Segundos de calentamiento

# =============================================================================
# GENERADORES DE DATOS
# =============================================================================
def generate_narma10(length, seed=42):
    np.random.seed(seed)
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])
        y[t] = np.clip(0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1, 0, 1)
    return u, y

# =============================================================================
# SERVIDOR STRATUM ASÍNCRONO (CORE)
# =============================================================================
class AsyncStratumServer(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Optimización de red crítica: Desactivar Nagle para mínima latencia
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(1)
        
        self.conn = None
        self.addr = None
        self.running = True
        self.authorized = False
        
        # Estructuras de datos concurrentes
        self.job_map = {}           # JobID -> {timestamp_sent, input_u, difficulty}
        self.share_queue = queue.Queue() # Shares recibidos (timestamp, job_id)
        self.job_counter = 0
        self.lock = threading.Lock()
        
    def run(self):
        print(f"[SERVER] Escuchando en {self.config.PORT} (Modo Async)...")
        try:
            self.conn, self.addr = self.sock.accept()
            print(f"[SERVER] Conectado: {self.addr}")
            self.conn.settimeout(0.1) # Non-blocking reads
            
            # Loop principal de lectura
            buffer = ""
            while self.running:
                try:
                    data = self.conn.recv(4096).decode('utf-8', errors='ignore')
                    if not data: break
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self._process_line(line)
                except (socket.timeout, ConnectionResetError):
                    continue
                except OSError as e:
                    if e.errno == 10038: # Socket closed
                        break
                    safe_print(f"[ERR] Loop: {e}")
                    break
                except Exception as e:
                    safe_print(f"[ERR] Loop: {e}")
                    break
        except Exception as e:
            safe_print(f"[ERR] Accept: {e}")

    def _process_line(self, line):
        try:
            msg = json.loads(line)
            method = msg.get('method')
            mid = msg.get('id')
            
            if method == 'mining.subscribe':
                self._send({"id": mid, "result": [[["mining.notify","1"]], "00", 4], "error": None})
                
            elif method == 'mining.authorize':
                self._send({"id": mid, "result": True, "error": None})
                self.authorized = True
                safe_print("[SERVER] Minero Autorizado. Iniciando protocolo de flujo.")
                
            elif method == 'mining.submit':
                # FÍSICA: Captura precisa del tiempo de llegada
                arrival_time = time.perf_counter()
                params = msg.get('params', [])
                # params: [worker, job_id, extranonce2, ntime, nonce]
                if len(params) >= 2:
                    job_id = params[1]
                    self.share_queue.put({
                        'job_id': job_id,
                        'arrival_time': arrival_time,
                        'nonce': params[4]
                    })
                    # Log discreto para ver que entran shares
                    if self.share_queue.qsize() % 10 == 0:
                        print(f"   [SHARE] Total: {self.share_queue.qsize()} (Last Job: {job_id})")
                self._send({"id": mid, "result": True, "error": None})
                
        except json.JSONDecodeError:
            pass

    def _send(self, data):
        try:
            if self.conn:
                self.conn.sendall((json.dumps(data) + '\n').encode())
        except: pass

    def send_work(self, u_value):
        """
        Envía un nuevo trabajo codificando el input u_value en la dificultad.
        NO ESPERA RESPUESTA. Fuego y olvido.
        """
        if not self.authorized: return None
        
        # Mapeo Input -> Dificultad (Rate Encoding)
        # u [0, 1] -> D [D_BASE, D_BASE + D_MOD]
        # Mayor input = Mayor dificultad = Menos shares (inhibición)
        # O viceversa, según se quiera modelar excitación/inhibición
        target_diff = self.config.D_BASE + (u_value * self.config.D_MODULATION)
        
        # 1. Ajustar dificultad
        self._send({"id": None, "method": "mining.set_difficulty", "params": [target_diff]})
        
        # 2. Enviar trabajo
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        # Crear timestamp de envío
        send_time = time.perf_counter()
        
        # Guardar en el mapa para correlación futura
        with self.lock:
            self.job_map[job_id] = {
                'u': u_value,
                'sent_time': send_time,
                'diff': target_diff
            }
            # Limpieza preventiva del mapa
            if len(self.job_map) > 5000:
                # Borrar los 1000 más viejos
                keys = list(self.job_map.keys())[:1000]
                for k in keys: del self.job_map[k]

        # Dummy coinbase params (válidos sintácticamente)
        coinb1 = "0100000001" + "0"*56
        ntime = hex(int(time.time()))[2:]
        
        self._send({
            "id": None, 
            "method": "mining.notify", 
            "params": [job_id, "0"*64, coinb1, "ffffffff", [], "20000000", "1f00ffff", ntime, True]
        })
        
        return job_id

    def stop(self):
        self.running = False
        if self.conn:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
                self.conn.close()
            except: pass
        self.sock.close()

def physical_reboot(ip):
    safe_print(f"\n[OS] Reiniciando minero en {ip}...")
    try:
        requests.post(f"http://{ip}/api/system/restart", timeout=5)
    except:
        pass
    time.sleep(45) # Esperar a que el minero vuelva y AxeOS esté listo
    safe_print("[OS] Minero listo.")

# =============================================================================
# EXPERIMENT ENGINE
# =============================================================================
def run_experiment(mode="normal"):
    config = Config()
    
    # 1. Generar Dataset
    safe_print(f"\n[GEN] Generando NARMA-10 ({config.NARMA_STEPS} pasos)...")
    u_data, y_target = generate_narma10(config.NARMA_STEPS + 100)
    # Recortar warmup
    u_data = u_data[100:]
    y_target = y_target[100:]
    
    if mode == "shuffle":
        np.random.shuffle(u_data) # Romper causalidad temporal
    elif mode == "constant":
        u_data = np.full_like(u_data, 0.25) # Input constante
        
    # 2. Iniciar Servidor
    server = AsyncStratumServer(config)
    server.start()
    
    safe_print("[WAIT] Esperando minero (conecta tu LV06 al puerto 3333)...")
    while not server.authorized:
        time.sleep(0.5)
        
    safe_print(f"[START] Iniciando inyección en modo: {mode.upper()}")
    safe_print(f"        Tasa de inyección: {config.STREAM_RATE_HZ} Hz")
    
    # 3. Bucle de Inyección (Streaming)
    # Este bucle corre a tiempo fijo, desacoplado de las respuestas
    
    results = [] # (input_u, target_y, features)
    
    # WARMUP FÍSICO
    safe_print("        Calentando silicio...")
    for _ in range(int(config.WARMUP_SEC * config.STREAM_RATE_HZ)):
        server.send_work(0.25)
        time.sleep(1.0 / config.STREAM_RATE_HZ)
        
    # BUCLE PRINCIPAL
    start_exp = time.time()
    for i, (u, y) in enumerate(zip(u_data, y_target)):
        # Inyectar
        job_id = server.send_work(u)
        
        # Esperar al siguiente ciclo (Rate Limiting preciso)
        time.sleep(1.0 / config.STREAM_RATE_HZ)
        
        if i % 100 == 0:
            safe_print(f"   Progreso: {i}/{len(u_data)} | Cola de shares: {server.share_queue.qsize()}")

    safe_print("[FIN] Inyección terminada. Esperando shares residuales...")
    time.sleep(2) # Esperar latencia de red final
    
    # 4. Procesamiento y Alineación (El secreto de V10)
    # Reconstruimos la historia correlacionando Job IDs
    
    safe_print("[PROCESS] Correlacionando inputs con outputs...")
    
    X = []
    Y = []
    
    # Extraemos todos los shares recibidos
    shares_list = []
    while not server.share_queue.empty():
        shares_list.append(server.share_queue.get())
        
    # Agrupamos shares por "Ventana de Tiempo" relativa al input
    # Como enviamos inputs a intervalos fijos, podemos usar ventanas de tiempo
    # O mejor: Agrupar por Job ID directamente.
    
    # Mapa inverso de JobID a índice de paso
    job_to_step = {} 
    # Recorremos el historial que guardó el servidor (accedemos a la estructura interna, es un hack permitido en PoC)
    # NOTA: En una implementación limpia, el servidor devolvería esto.
    
    # Construcción de vectores de características
    # Para cada paso 'i' del experimento (input u[i]):
    # Buscamos shares que correspondan al JobID enviado en ese paso O en los pasos cercanos (Efecto Memoria)
    
    # Estrategia simplificada V10: Ventana deslizante sobre shares recibidos
    # Alineamos por tiempo.
    
    sorted_shares = sorted(shares_list, key=lambda x: x['arrival_time'])
    
    # Convertir timestamps de shares a relativo
    if not sorted_shares:
        safe_print("ERROR CRÍTICO: No se recibieron shares. Revisa la dificultad o conexión.")
        server.stop()
        return None, None
        
    t0 = sorted_shares[0]['arrival_time']
    share_times = np.array([s['arrival_time'] for s in sorted_shares])
    
    # Reconstruimos X para cada paso de tiempo
    # Usamos la memoria del sistema: features = estadisticas de shares en los ultimos K segundos
    
    # Sabemos que enviamos inputs cada dt = 1/STREAM_RATE_HZ
    dt = 1.0 / config.STREAM_RATE_HZ
    
    # Estimamos el tiempo de inicio basado en los jobs (hack)
    # En producción usaríamos timestamps grabados en el bucle principal.
    # Asumimos que el primer share llega poco después del primer job.
    
    # Vamos a usar "Reservoir State Reconstruction"
    # Estado en tiempo t = Histograma de latencias de los últimos N shares
    
    # Ventana de integración del reservorio (memoria)
    mem_window = 0.5 # segundos
    
    # Iteramos sobre los pasos originales
    # Necesitamos reconstruir los timestamps de inyección
    # (En una versión real, los guardaríamos en una lista durante el bucle)
    # Simulamos timestamps ideales:
    
    # Re-alineación robusta:
    # Usamos el mapa de jobs del servidor para saber cuándo se envió cada U
    valid_steps = 0
    
    # Ordenar jobs por ID numérico
    sorted_jobs = sorted(
        [(k, v) for k, v in server.job_map.items()], 
        key=lambda x: int(x[0])
    )
    
    # Filtrar solo los jobs de este experimento (ignorando warmup)
    # Asumimos que los últimos N jobs corresponden a nuestro dataset
    experiment_jobs = sorted_jobs[-len(u_data):]
    
    for idx, (jid, info) in enumerate(experiment_jobs):
        t_sent = info['sent_time']
        u_val = info['u']
        target = y_target[idx]
        
        # Buscar shares que llegaron en la ventana [t_sent, t_sent + mem_window]
        # Esto captura la respuesta inmediata del chip
        
        relevant_shares = [
            s for s in sorted_shares 
            if t_sent <= s['arrival_time'] < (t_sent + mem_window)
        ]
        
        # Feature Extraction (HNS style - alta precisión)
        if len(relevant_shares) > 0:
            latencies = [s['arrival_time'] - t_sent for s in relevant_shares]
            
            # Features físicos
            count = len(latencies)
            mean_lat = np.mean(latencies) * 1000 # ms
            std_lat = np.std(latencies) * 1000   # ms
            min_lat = np.min(latencies) * 1000
            
            # HNS Proxy Features (simulando jerarquía)
            # Log-binning de latencias para capturar escalas
            
            f_vec = [
                count, 
                mean_lat, 
                std_lat, 
                min_lat,
                mean_lat / (count + 1), # Densidad inversa
                u_val # Feedback del input (opcional, ayuda a linealizar)
            ]
        else:
            # Estado de silencio (también es información)
            f_vec = [0, 0, 0, 0, 0, u_val]
            
        X.append(f_vec)
        Y.append(target)
        valid_steps += 1

    server.stop()
    return np.array(X), np.array(Y)

# =============================================================================
# ANÁLISIS
# =============================================================================
def evaluate_results(X, y):
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    # Metrics
    nmse = mean_squared_error(y_test, y_pred) / np.var(y_test)
    nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (np.max(y_test) - np.min(y_test))
    
    return nrmse

def main():
    safe_print("="*60)
    safe_print("   LV06 EXPERIMENT V10: SCALED STREAMING PROTOCOL")
    safe_print("   Validating Physical Reservoir Properties")
    safe_print("="*60)
    
    config = Config()
    
    # FASE 1: RE-REBOOT PARA LIMPIEZA
    physical_reboot(config.MINER_IP)
    
    safe_print("\n>>> FASE 1: REGIMEN NORMAL (Physical RC) <<<")
    X_norm, y_norm = run_experiment(mode="normal")
    if X_norm is None: return
    
    score_norm = evaluate_results(X_norm, y_norm)
    safe_print(f"\n   NRMSE (Normal): {score_norm:.4f}")
    
    # FASE 2: RE-REBOOT PARA SHUFFLE REAL
    physical_reboot(config.MINER_IP)
    
    safe_print("\n>>> FASE 2: REGIMEN SHUFFLE (Hardware-Level Control) <<<")
    # Ahora invocamos run_experiment con shuffle, lo que baraja u_data ANTES de inyectar
    # Esto valida si el ASIC reacciona a la CAUSALIDAD temporal de los datos.
    X_shuff, y_shuff = run_experiment(mode="shuffle")
    if X_shuff is None: return
    
    score_shuff = evaluate_results(X_shuff, y_shuff)
    safe_print(f"   NRMSE (Hardware Shuffle): {score_shuff:.4f}")
    
    safe_print("\n" + "="*60)
    safe_print("   VEREDICTO V10 REFINADO")
    safe_print("="*60)
    
    imp = (score_shuff - score_norm) / score_shuff * 100
    safe_print(f"Mejora sobre Hardware Shuffle: {imp:.2f}%")
    
    if imp > 10:
        safe_print("\n✅ ÉXITO CIENTÍFICO: El sistema muestra memoria física robusta.")
        safe_print("   El reservorio del ASIC es causal y temporalmente dependiente.")
    elif imp > 0:
        safe_print("\n⚠️ RESULTADO MARGINAL: Hay acoplamiento, pero la escala no es óptima.")
    else:
        safe_print("\n❌ FALLO: No se detecta diferencia entre causal y shuffle.")

if __name__ == "__main__":
    main()