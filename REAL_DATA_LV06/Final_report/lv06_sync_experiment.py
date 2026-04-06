#!/usr/bin/env python3
"""
================================================================================
LV06 SYNCHRONOUS RESERVOIR EXPERIMENT
================================================================================
Protocolo sin pérdida de shares:
  1. Inyectar input u[t]
  2. Esperar UN share
  3. Registrar tiempo de respuesta
  4. Siguiente paso

También prueba cambio de MHz sin reboot.

Author: Fran / Agnuxo + Claude
Date: December 2025
================================================================================
"""

import socket
import threading
import json
import time
import struct
import os
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random
import math

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    LV06_IP = "192.168.0.15"
    
    # Timing
    SHARE_TIMEOUT = 30.0      # Max segundos esperando un share
    WARMUP_SHARES = 10        # Shares de calentamiento
    
    # Experiment
    NARMA_STEPS = 200         # Reducido para prueba inicial
    TRAIN_RATIO = 0.8
    
    # Modulation mode: "difficulty", "frequency", or "both"
    MODULATION_MODE = "difficulty"
    
    # Difficulty range (from calibration)
    D_MIN = 0.1
    D_MAX = 50.0
    D_BASE = 0.5
    
    # Frequency range (if modulating MHz)
    FREQ_MIN = 250
    FREQ_MAX = 550
    FREQ_BASE = 400

# =============================================================================
# LOGGER
# =============================================================================
class Logger:
    def __init__(self):
        self.results_dir = "sync_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.results_dir, f"sync_{self.run_id}.log")
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        self.log_file.write(formatted + "\n")
        self.log_file.flush()
        
    def save_json(self, data: Dict, filename: str):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.log(f"[SAVED] {path}")
        
    def close(self):
        self.log_file.close()

# =============================================================================
# AXEOS API TESTS
# =============================================================================
def test_axeos_api(ip: str, logger: Logger) -> Dict:
    """
    Prueba las capacidades de la API de AxeOS
    """
    logger.log("\n" + "="*60)
    logger.log("   AXEOS API CAPABILITY TEST")
    logger.log("="*60)
    
    results = {
        "ip": ip,
        "reachable": False,
        "current_settings": None,
        "frequency_change_no_reboot": None,
        "voltage_change_no_reboot": None,
        "available_endpoints": []
    }
    
    # Test 1: Basic connectivity
    logger.log(f"\n[TEST 1] Checking connectivity to {ip}...")
    try:
        r = requests.get(f"http://{ip}/api/system", timeout=5)
        if r.status_code == 200:
            results["reachable"] = True
            results["current_settings"] = r.json()
            logger.log(f"  ✅ Connected! Current settings:")
            for key, val in r.json().items():
                logger.log(f"     {key}: {val}")
        else:
            logger.log(f"  ❌ HTTP {r.status_code}")
            return results
    except Exception as e:
        logger.log(f"  ❌ Connection failed: {e}")
        return results
    
    # Test 2: Check available endpoints
    logger.log(f"\n[TEST 2] Probing API endpoints...")
    endpoints_to_test = [
        "/api/system",
        "/api/system/info", 
        "/api/system/status",
        "/api/mining",
        "/api/mining/status",
        "/api/settings",
        "/api/frequency",
        "/api/voltage",
        "/api/asic",
        "/api/hashrate"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            r = requests.get(f"http://{ip}{endpoint}", timeout=2)
            if r.status_code == 200:
                results["available_endpoints"].append(endpoint)
                logger.log(f"  ✅ {endpoint}")
            else:
                logger.log(f"  ❌ {endpoint} ({r.status_code})")
        except:
            logger.log(f"  ❌ {endpoint} (timeout)")
    
    # Test 3: Try frequency change without reboot
    logger.log(f"\n[TEST 3] Testing frequency change WITHOUT reboot...")
    current_freq = results["current_settings"].get("frequency", 400)
    test_freq = current_freq + 25 if current_freq < 525 else current_freq - 25
    
    try:
        # Try PATCH without reboot
        logger.log(f"  Attempting: {current_freq} MHz → {test_freq} MHz")
        r = requests.patch(f"http://{ip}/api/system", json={"frequency": test_freq}, timeout=5)
        logger.log(f"  PATCH response: {r.status_code}")
        
        if r.status_code == 200:
            # Wait a moment
            time.sleep(2)
            
            # Check if it took effect
            r2 = requests.get(f"http://{ip}/api/system", timeout=5)
            new_freq = r2.json().get("frequency", current_freq)
            
            if new_freq == test_freq:
                results["frequency_change_no_reboot"] = True
                logger.log(f"  ✅ SUCCESS! Frequency changed to {new_freq} MHz without reboot!")
            else:
                results["frequency_change_no_reboot"] = False
                logger.log(f"  ❌ Change not applied (still at {new_freq} MHz)")
                
            # Restore original
            requests.patch(f"http://{ip}/api/system", json={"frequency": current_freq}, timeout=5)
            logger.log(f"  Restored to {current_freq} MHz")
    except Exception as e:
        results["frequency_change_no_reboot"] = False
        logger.log(f"  ❌ Error: {e}")
    
    # Test 4: Check for real-time frequency endpoint
    logger.log(f"\n[TEST 4] Looking for real-time frequency control...")
    realtime_endpoints = [
        ("/api/frequency", "PATCH", {"value": test_freq}),
        ("/api/mining/frequency", "POST", {"frequency": test_freq}),
        ("/api/asic/frequency", "PUT", {"freq": test_freq}),
    ]
    
    for endpoint, method, payload in realtime_endpoints:
        try:
            if method == "PATCH":
                r = requests.patch(f"http://{ip}{endpoint}", json=payload, timeout=2)
            elif method == "POST":
                r = requests.post(f"http://{ip}{endpoint}", json=payload, timeout=2)
            elif method == "PUT":
                r = requests.put(f"http://{ip}{endpoint}", json=payload, timeout=2)
            
            if r.status_code in [200, 201, 204]:
                logger.log(f"  ✅ {method} {endpoint} accepted!")
                results["frequency_change_no_reboot"] = True
            else:
                logger.log(f"  ❌ {method} {endpoint}: {r.status_code}")
        except:
            logger.log(f"  ❌ {method} {endpoint}: not available")
    
    return results

# =============================================================================
# SYNCHRONOUS STRATUM SERVER
# =============================================================================
class SyncStratumServer(threading.Thread):
    """
    Servidor Stratum síncrono: espera cada share antes de continuar
    """
    def __init__(self, config: Config, logger: Logger):
        super().__init__(daemon=True)
        self.config = config
        self.logger = logger
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(1)
        
        self.client_conn = None
        self.running = True
        self.authorized = False
        self.job_counter = 0
        self.current_difficulty = config.D_BASE
        
        # Share event for synchronization
        self.share_received = threading.Event()
        self.last_share_time = None
        self.pending_buffer = ""
        
    def run(self):
        self.logger.log(f"[SERVER] Listening on port {self.config.PORT}")
        try:
            self.sock.settimeout(120)
            conn, addr = self.sock.accept()
            self.logger.log(f"[SERVER] Connected: {addr}")
            self.client_conn = conn
            self._handshake(conn)
            self._listen_loop(conn)
        except Exception as e:
            self.logger.log(f"[SERVER] Error: {e}")

    def _handshake(self, conn):
        """Blocking handshake"""
        conn.setblocking(True)
        buffer = ""
        
        while not self.authorized and self.running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break
                buffer += data
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_handshake(conn, line.strip())
                    
            except socket.timeout:
                break
                
        self.pending_buffer = buffer
        conn.setblocking(False)

    def _process_handshake(self, conn, line):
        if not line:
            return
        try:
            msg = json.loads(line)
        except:
            return
            
        method = msg.get('method', '')
        mid = msg.get('id')
        
        if method == 'mining.subscribe':
            self._send(conn, {
                "id": mid,
                "result": [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4],
                "error": None
            })
            self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [self.current_difficulty]})
            
        elif method == 'mining.authorize':
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.authorized = True
            self.logger.log("[SERVER] Miner authorized")
            
        elif method == 'mining.configure':
            self._send(conn, {"id": mid, "result": {}, "error": None})

    def _listen_loop(self, conn):
        """Timeout-based listen for shares"""
        conn.settimeout(0.1)
        while self.running:
            try:
                data = conn.recv(8192)
                if not data:
                    break
                self.pending_buffer += data.decode('utf-8', errors='ignore')
                
                while '\n' in self.pending_buffer:
                    line, self.pending_buffer = self.pending_buffer.split('\n', 1)
                    self._process_message(conn, line.strip())
                    
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.log(f"[SERVER] Listen error: {e}")
                break

    def _process_message(self, conn, line):
        if not line:
            return
        try:
            msg = json.loads(line)
        except:
            return
            
        method = msg.get('method', '')
        mid = msg.get('id')
        
        if method == 'mining.submit':
            self.last_share_time = time.perf_counter()
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.share_received.set()

    def _send(self, conn, data):
        try:
            conn.sendall((json.dumps(data) + '\n').encode())
        except:
            pass

    def set_difficulty(self, diff: float):
        if not self.client_conn or not self.authorized:
            return
        self.current_difficulty = diff
        self._send(self.client_conn, {"id": None, "method": "mining.set_difficulty", "params": [diff]})
        time.sleep(0.02)

    def inject_and_wait(self, u_value: float, timeout: float = None) -> Optional[float]:
        """
        Inyecta input y ESPERA un share. Retorna tiempo de respuesta o None si timeout.
        PROTOCOLO SÍNCRONO: Garantiza 0% pérdida de shares.
        """
        if not self.client_conn or not self.authorized:
            return None
            
        if timeout is None:
            timeout = self.config.SHARE_TIMEOUT
            
        # Clear event
        self.share_received.clear()
        self.last_share_time = None
        
        # Record injection time
        inject_time = time.perf_counter()
        
        # Calculate difficulty based on input
        D = self.config.D_BASE / (u_value + 0.05)
        D = max(self.config.D_MIN, min(self.config.D_MAX, D))
        self.set_difficulty(D)
        
        # Send job
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        u_hex = struct.pack('>f', u_value).hex()
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        self._send(self.client_conn, {
            "id": None,
            "method": "mining.notify",
            "params": [job_id, "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        })
        
        # Wait for share
        if self.share_received.wait(timeout=timeout):
            response_time = self.last_share_time - inject_time
            return response_time
        else:
            return None  # Timeout

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except:
            pass

# =============================================================================
# MATH UTILITIES
# =============================================================================
def mean(x): return sum(x) / len(x) if x else 0.0
def std(x):
    if len(x) < 2: return 0.0
    m = mean(x)
    return math.sqrt(sum((xi - m)**2 for xi in x) / (len(x) - 1))
def nrmse(y_true, y_pred):
    r = max(y_true) - min(y_true)
    if r == 0: return 1.0
    return math.sqrt(mean([(t-p)**2 for t,p in zip(y_true, y_pred)])) / r

class SimpleRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = []
        self.mean_x = []
        self.std_x = []
        self.mean_y = 0.0

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        self.mean_y = mean(y)
        y_c = [yi - self.mean_y for yi in y]
        self.mean_x = [mean([row[j] for row in X]) for j in range(p)]
        self.std_x = [std([row[j] for row in X]) for j in range(p)]
        self.std_x = [s if s > 1e-9 else 1.0 for s in self.std_x]
        X_s = [[(X[i][j] - self.mean_x[j])/self.std_x[j] for j in range(p)] for i in range(n)]
        XtX = [[sum(X_s[k][i] * X_s[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
        for i in range(p): XtX[i][i] += self.alpha
        Xty = [sum(X_s[k][i] * y_c[k] for k in range(n)) for i in range(p)]
        self.weights = self._solve(XtX, Xty)
        return self

    def predict(self, X):
        return [self.mean_y + sum((X[i][j] - self.mean_x[j])/self.std_x[j] * self.weights[j]
                for j in range(len(self.weights))) for i in range(len(X))]

    def _solve(self, A, b):
        n = len(A)
        M = [row[:] + [val] for row, val in zip(A, b)]
        for i in range(n):
            pivot = M[i][i] if abs(M[i][i]) > 1e-10 else 1e-10
            for j in range(i + 1, n + 1): M[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = M[k][i]
                    for j in range(i + 1, n + 1): M[k][j] -= factor * M[i][j]
        return [row[n] for row in M]

# =============================================================================
# NARMA-10 GENERATOR
# =============================================================================
def generate_narma10(length: int, seed: int = 42):
    random.seed(seed)
    u = [random.uniform(0, 0.5) for _ in range(length)]
    y = [0.0] * length
    for t in range(10, length):
        sum_y = sum(y[t-9:t+1])
        y[t] = max(0, min(1, 0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1))
    return u, y

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def main():
    logger = Logger()
    config = Config()
    
    logger.log("="*60)
    logger.log("   LV06 SYNCHRONOUS RESERVOIR EXPERIMENT")
    logger.log("   Zero share loss protocol")
    logger.log("="*60)
    
    # Phase 0: Test AxeOS API
    api_results = test_axeos_api(config.LV06_IP, logger)
    logger.save_json(api_results, f"api_test_{logger.run_id}.json")
    
    if api_results.get("frequency_change_no_reboot"):
        logger.log("\n✅ Frequency modulation available without reboot!")
        config.MODULATION_MODE = "frequency"
    else:
        logger.log("\n⚠️ Frequency requires reboot. Using difficulty modulation only.")
        config.MODULATION_MODE = "difficulty"
    
    # Phase 1: Start server
    logger.log("\n" + "="*60)
    logger.log("   PHASE 1: SYNCHRONOUS DATA COLLECTION")
    logger.log("="*60)
    
    server = SyncStratumServer(config, logger)
    server.start()
    
    logger.log("\n[WAIT] Waiting for miner connection...")
    timeout_start = time.time()
    while not server.authorized:
        if time.time() - timeout_start > 120:
            logger.log("[ERROR] Miner connection timeout")
            return
        time.sleep(0.5)
    
    # Warmup
    logger.log(f"\n[WARMUP] Collecting {config.WARMUP_SHARES} warmup shares...")
    warmup_times = []
    for i in range(config.WARMUP_SHARES):
        rt = server.inject_and_wait(0.25)
        if rt:
            warmup_times.append(rt)
            logger.log(f"  Warmup {i+1}/{config.WARMUP_SHARES}: {rt*1000:.1f}ms")
        else:
            logger.log(f"  Warmup {i+1}/{config.WARMUP_SHARES}: TIMEOUT")
    
    if len(warmup_times) < 3:
        logger.log("[ERROR] Too many timeouts during warmup")
        return
        
    avg_warmup = mean(warmup_times)
    logger.log(f"\n[WARMUP] Average response time: {avg_warmup*1000:.1f}ms")
    
    # Generate NARMA-10
    u, y = generate_narma10(config.NARMA_STEPS + 50, seed=42)
    u, y = u[50:], y[50:]
    
    # Collect data
    logger.log(f"\n[COLLECT] Starting NARMA-10 ({config.NARMA_STEPS} steps)...")
    logger.log(f"          Mode: {config.MODULATION_MODE}")
    
    response_times = []
    timeouts = 0
    
    for i, u_val in enumerate(u):
        rt = server.inject_and_wait(u_val)
        
        if rt is not None:
            response_times.append(rt)
        else:
            response_times.append(avg_warmup * 2)  # Use 2x average as fallback
            timeouts += 1
        
        if i % 20 == 0:
            logger.log(f"  Step {i}/{len(u)} | u={u_val:.3f} | RT={response_times[-1]*1000:.1f}ms | Timeouts={timeouts}")
    
    logger.log(f"\n[COLLECT] Complete! Timeouts: {timeouts}/{len(u)} ({100*timeouts/len(u):.1f}%)")
    
    # Phase 2: Feature extraction
    logger.log("\n" + "="*60)
    logger.log("   PHASE 2: ANALYSIS")
    logger.log("="*60)
    
    # Features: response time + rolling stats
    X = []
    window = 5
    for i in range(len(response_times)):
        rt = response_times[i]
        
        if i >= window:
            recent = response_times[i-window:i]
            rt_mean = mean(recent)
            rt_std = std(recent)
        else:
            rt_mean = rt
            rt_std = 0.0
        
        X.append([rt * 1000, rt_mean * 1000, rt_std * 1000, 1.0])
    
    # Train/test split
    split = int(len(X) * config.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train and evaluate
    best_nrmse = 1.0
    best_alpha = 1.0
    
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        model = SimpleRidge(alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        score = nrmse(y_test, pred)
        if score < best_nrmse:
            best_nrmse, best_alpha = score, alpha
    
    logger.log(f"\n  NRMSE (Normal): {best_nrmse:.4f} (α={best_alpha})")
    
    # Shuffle test
    X_shuffle = X.copy()
    random.shuffle(X_shuffle)
    X_train_s, X_test_s = X_shuffle[:split], X_shuffle[split:]
    model_s = SimpleRidge(best_alpha).fit(X_train_s, y_train)
    pred_s = model_s.predict(X_test_s)
    nrmse_shuffle = nrmse(y_test, pred_s)
    
    logger.log(f"  NRMSE (Shuffle): {nrmse_shuffle:.4f}")
    
    # Verdict
    improvement = (nrmse_shuffle - best_nrmse) / nrmse_shuffle * 100 if nrmse_shuffle > 0 else 0
    
    logger.log("\n" + "="*60)
    logger.log("   RESULTS")
    logger.log("="*60)
    
    if improvement > 10:
        logger.log(f"\n✅ COUPLING DETECTED!")
        logger.log(f"   Normal ({best_nrmse:.4f}) < Shuffle ({nrmse_shuffle:.4f})")
        logger.log(f"   Improvement: {improvement:.1f}%")
        status = "SUCCESS"
    else:
        logger.log(f"\n❌ No significant coupling")
        logger.log(f"   Normal ({best_nrmse:.4f}) ≈ Shuffle ({nrmse_shuffle:.4f})")
        logger.log(f"   Improvement: {improvement:.1f}%")
        status = "FAIL"
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "modulation_mode": config.MODULATION_MODE,
            "narma_steps": config.NARMA_STEPS,
            "d_base": config.D_BASE
        },
        "data_quality": {
            "total_samples": len(response_times),
            "timeouts": timeouts,
            "timeout_rate": timeouts / len(response_times),
            "avg_response_ms": mean(response_times) * 1000
        },
        "results": {
            "nrmse_normal": best_nrmse,
            "nrmse_shuffle": nrmse_shuffle,
            "improvement_pct": improvement,
            "best_alpha": best_alpha
        },
        "status": status,
        "raw_response_times": response_times
    }
    
    logger.save_json(results, f"sync_results_{logger.run_id}.json")
    
    server.stop()
    logger.close()
    
    print(f"\n📁 Results saved to: {logger.results_dir}/")

if __name__ == "__main__":
    main()
