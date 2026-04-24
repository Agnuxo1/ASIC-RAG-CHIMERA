#!/usr/bin/env python3
"""
================================================================================
LV06 DEFINITIVE EXPERIMENT V6 - PERSISTENT LOGGING (SYNCHRONOUS PROTOCOL)
================================================================================
Protocolo Zero Share Loss:
  1. Inyectar input u[t]
  2. Esperar UN share
  3. Registrar tiempo de respuesta
  4. Siguiente paso

Archivos generados por cada run:
  - results/run_YYYYMMDD_HHMMSS.json     (resultados completos)
  - results/run_YYYYMMDD_HHMMSS.log      (log completo de consola)
  - results/HISTORY.jsonl                 (línea añadida, nunca sobrescrita)

Author: Fran / Agnuxo + Claude
Date: December 2025
================================================================================
"""

import socket
import threading
import json
import time
import struct
import random
import math
import os
import hashlib
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# =============================================================================
# PERSISTENT LOGGING SYSTEM
# =============================================================================
class ExperimentLogger:
    """
    Sistema de logging que NUNCA sobrescribe datos.
    Cada run tiene su propio archivo + se añade al historial.
    """
    
    def __init__(self):
        # Crear directorio de resultados
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Timestamp único para este run
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archivos para este run
        self.json_path = os.path.join(self.results_dir, f"run_{self.run_id}.json")
        self.log_path = os.path.join(self.results_dir, f"run_{self.run_id}.log")
        self.history_path = os.path.join(self.results_dir, "HISTORY.jsonl")
        
        # Abrir log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Buffer para todos los mensajes
        self.messages = []
        
        self.log(f"=== EXPERIMENT RUN {self.run_id} ===")
        self.log(f"Log file: {self.log_path}")
        self.log(f"Results file: {self.json_path}")
        
    def log(self, message: str):
        """Log mensaje a consola Y archivo"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"
        
        # Consola
        print(formatted)
        
        # Archivo
        self.log_file.write(formatted + "\n")
        self.log_file.flush()
        
        # Buffer
        self.messages.append(formatted)
        
    def save_results(self, results: Dict):
        """Guarda resultados en JSON único + añade al historial"""
        
        # Añadir metadata
        results['_meta'] = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'log_file': self.log_path,
            'code_hash': self._get_code_hash()
        }
        
        # 1. Guardar JSON completo (archivo único, nunca sobrescrito)
        with open(self.json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.log(f"[SAVED] Full results: {self.json_path}")
        
        # 2. Añadir línea al historial (append, nunca sobrescribe)
        summary = {
            'run_id': self.run_id,
            'timestamp': results['_meta']['timestamp'],
            'narma10_normal': results.get('narma10_normal'),
            'narma10_shuffle': results.get('narma10_shuffle'),
            'narma10_constant': results.get('narma10_constant'),
            'memory_capacity': results.get('memory_capacity'),
            'status': results.get('status')
        }
        
        with open(self.history_path, 'a') as f:
            f.write(json.dumps(summary) + "\n")
        self.log(f"[SAVED] History updated: {self.history_path}")
        
    def _get_code_hash(self) -> str:
        """Hash del código fuente para reproducibilidad"""
        try:
            with open(__file__, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            return "unknown"
            
    def close(self):
        self.log_file.close()

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    HOST: str = "0.0.0.0"
    PORT: int = 3333
    SHARE_TIMEOUT: float = 60.0  # Increased for multi-share
    HANDSHAKE_TIMEOUT: float = 120.0
    WARMUP_SHARES: int = 5
    D_BASE: float = 0.5        
    D_MIN: float = 0.1
    D_MAX: float = 50.0
    SHARES_PER_STEP: int = 3   # Law of Large Numbers: Reduce Poisson variance
    NARMA_STEPS: int = 100     # Reduced slightly to compensate for 3x shares
    MEMORY_STEPS: int = 100
    XOR_STEPS: int = 100
    TRAIN_RATIO: float = 0.8
    RIDGE_ALPHAS: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    RANDOM_SEED: int = 42

# =============================================================================
# MATH UTILITIES
# =============================================================================
def mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0

def variance(x: List[float]) -> float:
    if len(x) < 2: return 0.0
    m = mean(x)
    return sum((xi - m)**2 for xi in x) / (len(x) - 1)

def std(x: List[float]) -> float:
    return math.sqrt(variance(x))

def pearson_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 3: return 0.0
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx)**2 for xi in x))
    den_y = math.sqrt(sum((yi - my)**2 for yi in y))
    return num / (den_x * den_y) if (den_x * den_y) > 0 else 0.0

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(mean([(yt - yp)**2 for yt, yp in zip(y_true, y_pred)]))

def nrmse(y_true: List[float], y_pred: List[float]) -> float:
    y_range = max(y_true) - min(y_true)
    if y_range == 0: return 1.0
    return rmse(y_true, y_pred) / y_range

# =============================================================================
# RIDGE REGRESSION
# =============================================================================
class SimpleRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = []
        self.mean_x = []
        self.std_x = []
        self.mean_y = 0.0

    def fit(self, X: List[List[float]], y: List[float]):
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

    def predict(self, X: List[List[float]]) -> List[float]:
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
# TASK GENERATORS
# =============================================================================
def generate_narma10(length: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    random.seed(seed)
    u = [random.uniform(0, 0.5) for _ in range(length)]
    y = [0.0] * length
    for t in range(10, length):
        sum_y = sum(y[t-9:t+1])
        y[t] = max(0, min(1, 0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1))
    return u, y

def generate_memory_task(length: int, delay: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    random.seed(seed + delay)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0]*delay + u[:-delay]
    return u, y

def generate_xor_task(length: int, delay: int = 3, seed: int = 42) -> Tuple[List[float], List[float]]:
    random.seed(seed)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0] * length
    for t in range(delay, length):
        a, b = (1 if u[t] > 0.5 else 0), (1 if u[t-delay] > 0.5 else 0)
        y[t] = 1.0 if a != b else 0.0
    return u, y

# =============================================================================
# SYNC STRATUM SERVER
# =============================================================================
class SyncStratumServer(threading.Thread):
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
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
        self.share_received = threading.Event()
        self.collected_share_times = []
        self.target_share_count = 1
        self.injection_start_time = 0.0
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
        conn.setblocking(True)
        buffer = ""
        while not self.authorized and self.running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_handshake(conn, line.strip())
            except socket.timeout: break
        self.pending_buffer = buffer
        conn.setblocking(False)

    def _process_handshake(self, conn, line):
        if not line: return
        try: msg = json.loads(line)
        except: return
        method = msg.get('method', '')
        mid = msg.get('id')
        
        if method == 'mining.subscribe':
            self._send(conn, {"id": mid, "result": [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4], "error": None})
            self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [self.config.D_BASE]})
        elif method == 'mining.authorize':
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.authorized = True
            self.logger.log("[SERVER] Miner authorized")
        elif method == 'mining.configure':
            self._send(conn, {"id": mid, "result": {}, "error": None})

    def _listen_loop(self, conn):
        conn.settimeout(0.1)
        while self.running:
            try:
                data = conn.recv(8192)
                if not data: break
                self.pending_buffer += data.decode('utf-8', errors='ignore')
                while '\n' in self.pending_buffer:
                    line, self.pending_buffer = self.pending_buffer.split('\n', 1)
                    self._process_message(conn, line.strip())
            except socket.timeout: continue
            except Exception as e:
                self.logger.log(f"[SERVER] Listen error: {e}")
                break

    def _process_message(self, conn, line):
        if not line: return
        try: msg = json.loads(line)
        except: return
        method = msg.get('method', '')
        mid = msg.get('id')
        if method == 'mining.submit':
            now = time.perf_counter()
            self._send(conn, {"id": mid, "result": True, "error": None})
            
            # Logic for multi-share accumulation
            self.collected_share_times.append(now)
            if len(self.collected_share_times) >= self.target_share_count:
                self.share_received.set()

    def _send(self, conn, data):
        try: conn.sendall((json.dumps(data) + '\n').encode())
        except: pass

    def set_difficulty(self, diff: float):
        if not self.client_conn or not self.authorized: return
        self._send(self.client_conn, {"id": None, "method": "mining.set_difficulty", "params": [diff]})
        time.sleep(0.02)

    def inject_and_wait(self, u_value: float, timeout: float = None, shares_needed: int = 1) -> Optional[float]:
        if not self.client_conn or not self.authorized: return None
        if timeout is None: timeout = self.config.SHARE_TIMEOUT
        
        # Reset state for this injection
        self.share_received.clear()
        self.collected_share_times = []
        self.target_share_count = shares_needed
        self.injection_start_time = time.perf_counter()
        
        # Calculate and set difficulty
        D = self.config.D_BASE / (u_value + 0.05)
        D = max(self.config.D_MIN, min(self.config.D_MAX, D))
        self.set_difficulty(D)
        
        self.job_counter += 1
        u_hex = struct.pack('>f', u_value).hex()
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        self._send(self.client_conn, {
            "id": None, "method": "mining.notify",
            "params": [str(self.job_counter), "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        })
        
        # Wait for N shares
        if self.share_received.wait(timeout=timeout):
            # Return mean latency of the N collected shares
            latencies = [t - self.injection_start_time for t in self.collected_share_times]
            return mean(latencies)
        return None

    def stop(self):
        self.running = False
        try: self.sock.close()
        except: pass

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_task(server, logger, u_values, y_values, config, mode="normal"):
    logger.log(f"\n[TASK] Running mode: {mode.upper()}")
    response_times = []
    timeouts = 0
    
    # Pre-calculate fallback (avg of previous runs would be better, but assuming 5s here)
    fallback_time = 5.0 

    for i, u_val in enumerate(u_values):
        inject_val = 0.25 if mode == "constant" else u_val
        
        # Pass the configured SHARES_PER_STEP
        rt = server.inject_and_wait(inject_val, shares_needed=config.SHARES_PER_STEP)
        
        if rt is not None:
            response_times.append(rt)
        else:
            response_times.append(fallback_time * 2)
            timeouts += 1
        
        if i % 10 == 0:
            logger.log(f"  Step {i}/{len(u_values)} | u={inject_val:.3f} | RT={response_times[-1]*1000:.1f}ms | TO={timeouts}")

    logger.log(f"[TASK] {mode} Complete. Timeouts: {timeouts}/{len(u_values)} ({100*timeouts/len(u_values):.1f}%)")
    
    # Feature extraction (just simple RT and RT mean/std for now)
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
    
    if mode == "shuffle":
        random.shuffle(X)
        
    return X, y_values

def evaluate(X, y, config, logger) -> Tuple[float, float]:
    split = int(len(X) * config.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    best_nrmse, best_alpha = 1.0, 1.0
    best_pred = None
    for alpha in config.RIDGE_ALPHAS:
        model = SimpleRidge(alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        score = nrmse(y_test, pred)
        if score < best_nrmse:
            best_nrmse, best_alpha, best_pred = score, alpha, pred
    corr = pearson_correlation(y_test, best_pred) if best_pred else 0.0
    logger.log(f"  NRMSE = {best_nrmse:.4f} | r = {corr:.4f} | α = {best_alpha}")
    return best_nrmse, corr

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = ExperimentLogger()
    config = ExperimentConfig()
    
    logger.log("=" * 60)
    logger.log("   LV06 DEFINITIVE EXPERIMENT V6 - SYNC PROTOCOL")
    logger.log("=" * 60)
    logger.log(f"Config: D_BASE={config.D_BASE}, TIMEOUT={config.SHARE_TIMEOUT}s, STEPS={config.NARMA_STEPS}")
    
    results = {
        'config': asdict(config),
        'hardware': {'device': 'LV06', 'protocol': 'Sync RC'}
    }
    
    server = SyncStratumServer(config, logger)
    server.start()
    
    logger.log("\n[WAIT] Waiting for miner connection...")
    timeout_start = time.time()
    while not server.authorized:
        if time.time() - timeout_start > config.HANDSHAKE_TIMEOUT:
            logger.log("[ERROR] Miner connection timeout")
            results['status'] = 'TIMEOUT'
            logger.save_results(results)
            return
        time.sleep(0.5)
    
    # Warmup
    logger.log(f"\n[WARMUP] {config.WARMUP_SHARES} shares...")
    warmup_times = []
    for _ in range(config.WARMUP_SHARES):
        rt = server.inject_and_wait(0.25)
        if rt: warmup_times.append(rt)
    
    if not warmup_times:
        logger.log("[ERROR] Warmup failed (no shares)")
        return
        
    # Generate NARMA-10 data
    u, y = generate_narma10(config.NARMA_STEPS + 50, config.RANDOM_SEED)
    u, y = u[50:], y[50:]
    
    # === PHASE 1: NARMA-10 BATTERY ===
    logger.log("\n" + "=" * 60)
    logger.log("   PHASE 1: NARMA-10 BENCHMARK")
    logger.log("=" * 60)
    
    X_normal, _ = run_task(server, logger, u, y, config, mode="normal")
    nrmse_normal, corr_normal = evaluate(X_normal, y, config, logger)
    results['narma10_normal'] = nrmse_normal
    results['narma10_normal_corr'] = corr_normal
    
    X_shuffle, _ = run_task(server, logger, u, y, config, mode="shuffle")
    nrmse_shuffle, corr_shuffle = evaluate(X_shuffle, y, config, logger)
    results['narma10_shuffle'] = nrmse_shuffle
    results['narma10_shuffle_corr'] = corr_shuffle
    
    X_constant, _ = run_task(server, logger, u, y, config, mode="constant")
    nrmse_constant, corr_constant = evaluate(X_constant, y, config, logger)
    results['narma10_constant'] = nrmse_constant
    results['narma10_constant_corr'] = corr_constant
    
    # === PHASE 2: MEMORY CAPACITY ===
    logger.log("\n" + "=" * 60)
    logger.log("   PHASE 2: MEMORY CAPACITY")
    logger.log("=" * 60)
    
    mc_scores = []
    for delay in [1, 2, 3, 5]:
        u_mem, y_mem = generate_memory_task(config.MEMORY_STEPS + 50, delay, config.RANDOM_SEED)
        u_mem, y_mem = u_mem[50:], y_mem[50:]
        X_mem, _ = run_task(server, logger, u_mem, y_mem, config, mode="normal")
        _, corr = evaluate(X_mem, y_mem, config, logger)
        mc = max(0, corr ** 2)
        mc_scores.append((delay, mc))
        logger.log(f"  Delay {delay}: MC = {mc:.4f}")
    
    total_mc = sum(mc for _, mc in mc_scores)
    results['memory_capacity'] = total_mc
    results['memory_scores'] = mc_scores
    logger.log(f"\n  TOTAL MEMORY CAPACITY: {total_mc:.4f}")
    
    # === PHASE 3: XOR NONLINEARITY ===
    logger.log("\n" + "=" * 60)
    logger.log("   PHASE 3: XOR NONLINEARITY TEST")
    logger.log("=" * 60)
    
    u_xor, y_xor = generate_xor_task(config.XOR_STEPS + 50, delay=3, seed=config.RANDOM_SEED)
    u_xor, y_xor = u_xor[50:], y_xor[50:]
    X_xor, _ = run_task(server, logger, u_xor, y_xor, config, mode="normal")
    
    split = int(len(X_xor) * config.TRAIN_RATIO)
    model = SimpleRidge(1.0).fit(X_xor[:split], y_xor[:split])
    pred = model.predict(X_xor[split:])
    accuracy = mean([1.0 if (p > 0.5) == (t > 0.5) else 0.0 for p, t in zip(pred, y_xor[split:])])
    results['xor_accuracy'] = accuracy
    logger.log(f"  XOR Accuracy: {accuracy:.2%} (baseline: 50%)")
    
    # === VERDICTS ===
    logger.log("\n" + "=" * 60)
    logger.log("   VERDICTS")
    logger.log("=" * 60)
    
    verdicts = {}
    
    # Coupling test
    if nrmse_shuffle > 0:
        improvement = (nrmse_shuffle - nrmse_normal) / nrmse_shuffle * 100
        passed = improvement > 10
        verdicts['coupling'] = {'passed': passed, 'improvement': improvement}
        symbol = "✅" if passed else "❌"
        logger.log(f"{symbol} COUPLING: Normal ({nrmse_normal:.4f}) vs Shuffle ({nrmse_shuffle:.4f}) = {improvement:.1f}% improvement")
    
    # Entropy test
    if nrmse_constant > 0:
        improvement = (nrmse_constant - nrmse_normal) / nrmse_constant * 100
        passed = improvement > 10
        verdicts['entropy'] = {'passed': passed, 'improvement': improvement}
        symbol = "✅" if passed else "❌"
        logger.log(f"{symbol} ENTROPY: Normal ({nrmse_normal:.4f}) vs Constant ({nrmse_constant:.4f}) = {improvement:.1f}% improvement")
    
    # Memory test
    passed = total_mc > 0.5
    verdicts['memory'] = {'passed': passed, 'value': total_mc}
    symbol = "✅" if passed else "❌"
    logger.log(f"{symbol} MEMORY: MC = {total_mc:.4f} (threshold: 0.5)")
    
    # XOR test
    passed = accuracy > 0.6
    verdicts['nonlinearity'] = {'passed': passed, 'value': accuracy}
    symbol = "✅" if passed else "❌"
    logger.log(f"{symbol} NONLINEARITY: Accuracy = {accuracy:.2%} (threshold: 60%)")
    
    results['verdicts'] = verdicts
    
    # === FINAL STATUS ===
    passed_count = sum(1 for v in verdicts.values() if v['passed'])
    total_count = len(verdicts)
    
    if passed_count == total_count:
        status = "✅ GENUINE RESERVOIR COMPUTER"
    elif verdicts.get('coupling', {}).get('passed') and verdicts.get('entropy', {}).get('passed'):
        status = "⚠️ RATE ENCODER (coupling verified, but limited RC properties)"
    else:
        status = "❌ NO RC CAPABILITY DEMONSTRATED"
    
    results['status'] = status
    results['passed'] = passed_count
    results['total'] = total_count
    
    logger.log("\n" + "=" * 60)
    logger.log(f"   FINAL STATUS: {status}")
    logger.log(f"   Tests passed: {passed_count}/{total_count}")
    logger.log("=" * 60)
    
    # Save everything
    logger.save_results(results)
    
    server.stop()
    logger.close()
    
    print(f"\n📁 All files saved in: {logger.results_dir}/")
    print(f"   - {logger.json_path}")
    print(f"   - {logger.log_path}")
    print(f"   - {logger.history_path}")

if __name__ == "__main__":
    main()
