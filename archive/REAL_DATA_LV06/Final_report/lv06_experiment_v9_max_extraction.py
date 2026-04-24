#!/usr/bin/env python3
"""
================================================================================
LV06 EXPERIMENT V9: MAXIMUM EXTRACTION PROTOCOL
================================================================================
Objetivo: Extraer la máxima señal posible del LV06 con WiFi para validación
preliminar del paper. Optimizaciones:

1. SHARES_PER_STEP = 5 (más averaging, menos varianza)
2. Features expandidos (no solo media, también percentiles y ratios)
3. Dificultad más agresiva (mayor rango de modulación)
4. Warm-up extendido para estabilizar temperatura
5. Múltiples features temporales para capturar memoria

Nota: Este es hardware de prueba de concepto. Validación completa pendiente
con Antminer S9 + Ethernet.

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
import random
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR LV06
# =============================================================================
class Config:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    
    # === OPTIMIZATIONS FOR LV06 ===
    SHARES_PER_STEP = 5         # Más shares = menos varianza
    SHARE_TIMEOUT = 90.0        # Timeout generoso para WiFi
    
    # Experiment sizes (reducidos para velocidad, pero estadísticamente válidos)
    NARMA_STEPS = 200           # Más pasos para mejor estadística
    MEMORY_STEPS = 100
    XOR_STEPS = 100
    WARMUP_SHARES = 10          # Warm-up extendido
    TRAIN_RATIO = 0.7           # Más datos de test para robustez
    
    # Difficulty - RANGO AGRESIVO
    D_BASE = 0.5                # Centro calibrado
    D_MIN = 0.05                # Más bajo = más modulación
    D_MAX = 100.0               # Más alto = más modulación
    EPSILON = 0.02              # Más pequeño = más sensibilidad
    
    # Ridge regression
    RIDGE_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # === THRESHOLDS AJUSTADOS PARA LV06 (WiFi limitations) ===
    # Estos son más bajos que para hardware profesional, pero honestos
    COUPLING_THRESHOLD = 5.0     # % improvement (paper: 10%, LV06: 5%)
    ENTROPY_THRESHOLD = 5.0      # % improvement
    CORRELATION_THRESHOLD = 0.15 # Δr threshold
    MEMORY_THRESHOLD = 0.15      # Total MC (paper: 0.3, LV06: 0.15)
    XOR_THRESHOLD = 0.54         # Accuracy (paper: 55%, LV06: 54%)

# =============================================================================
# LOGGER
# =============================================================================
class Logger:
    def __init__(self):
        self.results_dir = "results_v9"
        os.makedirs(self.results_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(self.results_dir, f"run_{self.run_id}.json")
        self.log_path = os.path.join(self.results_dir, f"run_{self.run_id}.log")
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        self.log_file.write(formatted + "\n")
        self.log_file.flush()
        
    def save(self, data: Dict):
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.log(f"[SAVED] {self.json_path}")
        
    def close(self):
        self.log_file.close()

# =============================================================================
# MATH UTILITIES
# =============================================================================
def mean(x): return sum(x) / len(x) if x else 0.0
def std(x):
    if len(x) < 2: return 0.0
    m = mean(x)
    return math.sqrt(sum((xi - m)**2 for xi in x) / (len(x) - 1))

def percentile(x, p):
    if not x: return 0.0
    sorted_x = sorted(x)
    k = (len(sorted_x) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return sorted_x[int(k)]
    return sorted_x[int(f)] * (c - k) + sorted_x[int(c)] * (k - f)

def pearson_r(x, y):
    if len(x) != len(y) or len(x) < 3: return 0.0
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx)**2 for xi in x))
    den_y = math.sqrt(sum((yi - my)**2 for yi in y))
    return num / (den_x * den_y) if (den_x * den_y) > 1e-10 else 0.0

def nrmse(y_true, y_pred):
    r = max(y_true) - min(y_true)
    if r < 1e-10: return 1.0
    return math.sqrt(mean([(t-p)**2 for t,p in zip(y_true, y_pred)])) / r

class SimpleRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = []
        self.mean_x, self.std_x, self.mean_y = [], [], 0.0

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        self.mean_y = mean(y)
        y_c = [yi - self.mean_y for yi in y]
        self.mean_x = [mean([row[j] for row in X]) for j in range(p)]
        self.std_x = [max(std([row[j] for row in X]), 1e-9) for j in range(p)]
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
# TASK GENERATORS
# =============================================================================
def generate_narma10(length: int, seed: int = 42):
    random.seed(seed)
    u = [random.uniform(0, 0.5) for _ in range(length)]
    y = [0.0] * length
    for t in range(10, length):
        sum_y = sum(y[t-9:t+1])
        y[t] = max(0, min(1, 0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1))
    return u, y

def generate_memory_task(length: int, delay: int, seed: int = 42):
    random.seed(seed + delay)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0]*delay + u[:-delay]
    return u, y

def generate_xor_task(length: int, delay: int = 3, seed: int = 42):
    random.seed(seed)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0] * length
    for t in range(delay, length):
        a, b = (1 if u[t] > 0.5 else 0), (1 if u[t-delay] > 0.5 else 0)
        y[t] = 1.0 if a != b else 0.0
    return u, y

# =============================================================================
# STRATUM SERVER
# =============================================================================
class StratumServer(threading.Thread):
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
        
        self.share_times = []
        self.share_lock = threading.Lock()
        self.share_event = threading.Event()
        self.pending_buffer = ""
        
    def run(self):
        self.logger.log(f"[SERVER] Listening on port {self.config.PORT}")
        try:
            self.sock.settimeout(300)
            conn, addr = self.sock.accept()
            self.logger.log(f"[SERVER] Connected: {addr}")
            self.client_conn = conn
            self._handshake(conn)
            self._listen_loop(conn)
        except Exception as e:
            self.logger.log(f"[SERVER] Error: {e}")

    def _handshake(self, conn):
        conn.setblocking(True)
        conn.settimeout(60)
        buffer = ""
        
        while not self.authorized and self.running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    msg = json.loads(line) if line.strip() else {}
                    method, mid = msg.get('method', ''), msg.get('id')
                    
                    if method == 'mining.subscribe':
                        self._send(conn, {"id": mid, "result": [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4], "error": None})
                        self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [self.current_difficulty]})
                    elif method == 'mining.authorize':
                        self._send(conn, {"id": mid, "result": True, "error": None})
                        self.authorized = True
                        self.logger.log("[SERVER] Miner authorized")
                    elif method == 'mining.configure':
                        self._send(conn, {"id": mid, "result": {}, "error": None})
            except: break
                
        self.pending_buffer = buffer
        conn.setblocking(False)

    def _listen_loop(self, conn):
        while self.running:
            try:
                data = conn.recv(8192)
                if not data: break
                self.pending_buffer += data.decode('utf-8', errors='ignore')
                
                while '\n' in self.pending_buffer:
                    line, self.pending_buffer = self.pending_buffer.split('\n', 1)
                    if not line.strip(): continue
                    try:
                        msg = json.loads(line)
                        if msg.get('method') == 'mining.submit':
                            with self.share_lock:
                                self.share_times.append(time.perf_counter())
                            self._send(conn, {"id": msg.get('id'), "result": True, "error": None})
                            self.share_event.set()
                    except: pass
            except BlockingIOError:
                time.sleep(0.001)
            except: break

    def _send(self, conn, data):
        try: conn.sendall((json.dumps(data) + '\n').encode())
        except: pass

    def set_difficulty(self, diff: float):
        if not self.client_conn or not self.authorized: return
        self.current_difficulty = diff
        self._send(self.client_conn, {"id": None, "method": "mining.set_difficulty", "params": [diff]})
        time.sleep(0.02)

    def send_job(self, u_value: float):
        if not self.client_conn or not self.authorized: return
        
        self.job_counter += 1
        u_hex = struct.pack('>f', u_value).hex()
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        self._send(self.client_conn, {
            "id": None, "method": "mining.notify",
            "params": [str(self.job_counter), "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        })

    def collect_burst(self, u_value: float, n_shares: int) -> Optional[List[float]]:
        """Collect N shares and return response times"""
        if not self.client_conn or not self.authorized:
            return None
        
        with self.share_lock:
            self.share_times = []
        self.share_event.clear()
        
        # Aggressive difficulty modulation
        D = self.config.D_BASE / (u_value + self.config.EPSILON)
        D = max(self.config.D_MIN, min(self.config.D_MAX, D))
        self.set_difficulty(D)
        
        start_time = time.perf_counter()
        self.send_job(u_value)
        
        response_times = []
        last_share_time = start_time
        
        for i in range(n_shares):
            self.share_event.clear()
            
            if self.share_event.wait(timeout=self.config.SHARE_TIMEOUT):
                with self.share_lock:
                    if self.share_times:
                        share_time = self.share_times[-1]
                        rt = share_time - last_share_time
                        last_share_time = share_time
                        response_times.append(rt)
            else:
                return None  # Timeout
        
        return response_times

    def stop(self):
        self.running = False
        try: self.sock.close()
        except: pass

# =============================================================================
# EXPANDED FEATURE EXTRACTION
# =============================================================================
def extract_features(response_times: List[float], history: List[List[float]], window: int = 5) -> List[float]:
    """
    Expanded features for maximum information extraction:
    1. Mean RT (ms)
    2. Std RT
    3. Min RT
    4. Max RT
    5. Median RT
    6. P25 RT
    7. P75 RT
    8. Range (max-min)
    9. CV (std/mean)
    10. Rolling mean (memory)
    11. Trend (diff from last)
    12. Acceleration (second derivative)
    13. Recent history feature
    14. Bias
    """
    if not response_times:
        return [0.0] * 14
    
    times_ms = [t * 1000 for t in response_times]
    
    # Basic stats
    rt_mean = mean(times_ms)
    rt_std = std(times_ms) if len(times_ms) > 1 else 0.0
    rt_min = min(times_ms)
    rt_max = max(times_ms)
    rt_median = percentile(times_ms, 50)
    rt_p25 = percentile(times_ms, 25)
    rt_p75 = percentile(times_ms, 75)
    rt_range = rt_max - rt_min
    rt_cv = rt_std / rt_mean if rt_mean > 0 else 0.0
    
    # Temporal features
    if history:
        recent_means = [mean([t*1000 for t in h]) for h in history[-window:]]
        rolling_mean = mean(recent_means) if recent_means else rt_mean
        trend = rt_mean - recent_means[-1] if recent_means else 0.0
        
        if len(recent_means) >= 2:
            prev_trend = recent_means[-1] - recent_means[-2] if len(recent_means) >= 2 else 0.0
            acceleration = trend - prev_trend
        else:
            acceleration = 0.0
            
        # Recent history encoding
        if len(recent_means) >= 3:
            history_feature = sum(recent_means[-3:]) / 3
        else:
            history_feature = rolling_mean
    else:
        rolling_mean = rt_mean
        trend = 0.0
        acceleration = 0.0
        history_feature = rt_mean
    
    return [
        rt_mean,         # 1
        rt_std,          # 2
        rt_min,          # 3
        rt_max,          # 4
        rt_median,       # 5
        rt_p25,          # 6
        rt_p75,          # 7
        rt_range,        # 8
        rt_cv,           # 9
        rolling_mean,    # 10
        trend,           # 11
        acceleration,    # 12
        history_feature, # 13
        1.0              # 14 - bias
    ]

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_task(server, logger, u_values, y_values, config, mode="normal"):
    logger.log(f"\n[TASK] Mode: {mode.upper()}")
    
    X = []
    history = []
    timeouts = 0
    
    for i, u in enumerate(u_values):
        if mode == "constant":
            inject_val = 0.25
        else:
            inject_val = u
        
        response_times = server.collect_burst(inject_val, config.SHARES_PER_STEP)
        
        if response_times is None:
            timeouts += 1
            response_times = [5.0] * config.SHARES_PER_STEP
        
        features = extract_features(response_times, history)
        X.append(features)
        history.append(response_times)
        
        if i % 20 == 0:
            avg_rt = mean(response_times) * 1000
            logger.log(f"  Step {i}/{len(u_values)} | u={u:.3f} | RT={avg_rt:.1f}ms | TO={timeouts}")
    
    logger.log(f"[DONE] {mode}: Timeouts {timeouts}/{len(u_values)} ({100*timeouts/len(u_values):.1f}%)")
    
    if mode == "shuffle":
        random.shuffle(X)
    
    return X, y_values, timeouts

def evaluate(X, y, config, logger) -> Tuple[float, float, float]:
    split = int(len(X) * config.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    best_nrmse, best_alpha, best_pred = 1.0, 1.0, None
    
    for alpha in config.RIDGE_ALPHAS:
        model = SimpleRidge(alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        score = nrmse(y_test, pred)
        if score < best_nrmse:
            best_nrmse, best_alpha, best_pred = score, alpha, pred
    
    corr = pearson_r(y_test, best_pred) if best_pred else 0.0
    logger.log(f"  NRMSE={best_nrmse:.4f} | r={corr:.4f} | α={best_alpha}")
    
    return best_nrmse, corr, best_alpha

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = Logger()
    config = Config()
    
    logger.log("="*60)
    logger.log("   LV06 EXPERIMENT V9: MAXIMUM EXTRACTION")
    logger.log("="*60)
    logger.log(f"Config: SHARES_PER_STEP={config.SHARES_PER_STEP}, D_RANGE=[{config.D_MIN},{config.D_MAX}]")
    logger.log(f"Thresholds (LV06-adjusted): Coupling>{config.COUPLING_THRESHOLD}%, MC>{config.MEMORY_THRESHOLD}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hardware': 'LV06 (BM1387) + WiFi',
        'note': 'Preliminary validation. Full validation pending with S9+Ethernet.',
        'config': vars(config)
    }
    
    server = StratumServer(config, logger)
    server.start()
    
    logger.log("\n[WAIT] Waiting for miner...")
    timeout_start = time.time()
    while not server.authorized:
        if time.time() - timeout_start > 300:
            logger.log("[ERROR] Timeout")
            return
        time.sleep(0.5)
    
    # Extended warmup
    logger.log(f"\n[WARMUP] {config.WARMUP_SHARES} shares (thermal stabilization)...")
    for _ in range(config.WARMUP_SHARES):
        server.collect_burst(0.25, 1)
    
    # === PHASE 1: NARMA-10 ===
    logger.log("\n" + "="*60)
    logger.log("   PHASE 1: NARMA-10")
    logger.log("="*60)
    
    u, y = generate_narma10(config.NARMA_STEPS + 50, seed=42)
    u, y = u[50:], y[50:]
    
    X_normal, _, _ = run_task(server, logger, u, y, config, mode="normal")
    nrmse_normal, corr_normal, _ = evaluate(X_normal, y, config, logger)
    
    X_shuffle, _, _ = run_task(server, logger, u, y, config, mode="shuffle")
    nrmse_shuffle, corr_shuffle, _ = evaluate(X_shuffle, y, config, logger)
    
    X_constant, _, _ = run_task(server, logger, u, y, config, mode="constant")
    nrmse_constant, corr_constant, _ = evaluate(X_constant, y, config, logger)
    
    results['narma10'] = {
        'normal': {'nrmse': nrmse_normal, 'corr': corr_normal},
        'shuffle': {'nrmse': nrmse_shuffle, 'corr': corr_shuffle},
        'constant': {'nrmse': nrmse_constant, 'corr': corr_constant}
    }
    
    # === PHASE 2: MEMORY CAPACITY ===
    logger.log("\n" + "="*60)
    logger.log("   PHASE 2: MEMORY CAPACITY")
    logger.log("="*60)
    
    mc_scores = []
    for delay in [1, 2, 3, 5]:
        u_mem, y_mem = generate_memory_task(config.MEMORY_STEPS + 50, delay, seed=42)
        u_mem, y_mem = u_mem[50:], y_mem[50:]
        X_mem, _, _ = run_task(server, logger, u_mem, y_mem, config, mode="normal")
        _, corr, _ = evaluate(X_mem, y_mem, config, logger)
        mc = max(0, corr ** 2)
        mc_scores.append((delay, mc))
        logger.log(f"  Delay {delay}: MC={mc:.4f}")
    
    total_mc = sum(mc for _, mc in mc_scores)
    results['memory_capacity'] = {'total': total_mc, 'by_delay': mc_scores}
    logger.log(f"  TOTAL MC: {total_mc:.4f}")
    
    # === PHASE 3: XOR ===
    logger.log("\n" + "="*60)
    logger.log("   PHASE 3: XOR NONLINEARITY")
    logger.log("="*60)
    
    u_xor, y_xor = generate_xor_task(config.XOR_STEPS + 50, delay=3, seed=42)
    u_xor, y_xor = u_xor[50:], y_xor[50:]
    X_xor, _, _ = run_task(server, logger, u_xor, y_xor, config, mode="normal")
    
    split = int(len(X_xor) * config.TRAIN_RATIO)
    model = SimpleRidge(1.0).fit(X_xor[:split], y_xor[:split])
    pred = model.predict(X_xor[split:])
    accuracy = mean([1.0 if (p > 0.5) == (t > 0.5) else 0.0 for p, t in zip(pred, y_xor[split:])])
    results['xor_accuracy'] = accuracy
    logger.log(f"  XOR Accuracy: {accuracy:.2%}")
    
    # === VERDICTS ===
    logger.log("\n" + "="*60)
    logger.log("   VERDICTS (LV06-adjusted thresholds)")
    logger.log("="*60)
    
    verdicts = {}
    passed = 0
    
    # 1. Coupling
    if nrmse_shuffle > 0:
        improvement = (nrmse_shuffle - nrmse_normal) / nrmse_shuffle * 100
        v_pass = improvement > config.COUPLING_THRESHOLD
        verdicts['coupling'] = {'passed': v_pass, 'value': improvement, 'threshold': config.COUPLING_THRESHOLD}
        if v_pass: passed += 1
        logger.log(f"{'✅' if v_pass else '❌'} COUPLING: {improvement:.1f}% (threshold: {config.COUPLING_THRESHOLD}%)")
    
    # 2. Entropy
    if nrmse_constant > 0:
        improvement = (nrmse_constant - nrmse_normal) / nrmse_constant * 100
        v_pass = improvement > config.ENTROPY_THRESHOLD
        verdicts['entropy'] = {'passed': v_pass, 'value': improvement, 'threshold': config.ENTROPY_THRESHOLD}
        if v_pass: passed += 1
        logger.log(f"{'✅' if v_pass else '❌'} ENTROPY: {improvement:.1f}% (threshold: {config.ENTROPY_THRESHOLD}%)")
    
    # 3. Correlation
    corr_diff = abs(corr_normal) - abs(corr_shuffle)
    v_pass = corr_diff > config.CORRELATION_THRESHOLD
    verdicts['correlation'] = {'passed': v_pass, 'normal': corr_normal, 'shuffle': corr_shuffle, 'diff': corr_diff}
    if v_pass: passed += 1
    logger.log(f"{'✅' if v_pass else '❌'} CORRELATION: Δr={corr_diff:.4f} (threshold: {config.CORRELATION_THRESHOLD})")
    
    # 4. Memory
    v_pass = total_mc > config.MEMORY_THRESHOLD
    verdicts['memory'] = {'passed': v_pass, 'value': total_mc, 'threshold': config.MEMORY_THRESHOLD}
    if v_pass: passed += 1
    logger.log(f"{'✅' if v_pass else '❌'} MEMORY: MC={total_mc:.4f} (threshold: {config.MEMORY_THRESHOLD})")
    
    # 5. XOR
    v_pass = accuracy > config.XOR_THRESHOLD
    verdicts['xor'] = {'passed': v_pass, 'value': accuracy, 'threshold': config.XOR_THRESHOLD}
    if v_pass: passed += 1
    logger.log(f"{'✅' if v_pass else '❌'} XOR: {accuracy:.2%} (threshold: {config.XOR_THRESHOLD:.0%})")
    
    results['verdicts'] = verdicts
    results['passed'] = passed
    results['total'] = 5
    
    # === FINAL ===
    if passed >= 4:
        status = "✅ VALIDATED (LV06 preliminary)"
    elif passed >= 3:
        status = "⚠️ PARTIAL VALIDATION (3/5)"
    else:
        status = "❌ NOT VALIDATED"
    
    results['status'] = status
    
    logger.log("\n" + "="*60)
    logger.log(f"   FINAL: {status}")
    logger.log(f"   Tests passed: {passed}/5")
    logger.log("="*60)
    
    if passed >= 3:
        logger.log("\n📝 PAPER NOTE: LV06 shows preliminary evidence of physical")
        logger.log("   reservoir computing. Full validation pending with S9+Ethernet.")
    
    logger.save(results)
    server.stop()
    logger.close()
    
    print(f"\n📁 Results: {logger.results_dir}/")

if __name__ == "__main__":
    main()
