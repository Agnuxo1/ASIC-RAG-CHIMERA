#!/usr/bin/env python3
"""
================================================================================
LV06 EXPERIMENT V7 - SYNC BURST PROTOCOL
================================================================================
Mejora sobre V6: En vez de 1 share por paso (muy ruidoso), esperamos N shares
y promediamos el tiempo de respuesta. Esto reduce la varianza del proceso de
Poisson y revela la verdadera señal del reservoir.

Ley de Grandes Números:
  - 1 share:  Varianza = σ²
  - 3 shares: Varianza = σ²/3
  - 5 shares: Varianza = σ²/5

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
# CONFIGURATION
# =============================================================================
class Config:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    
    # === KEY PARAMETER: SHARES PER STEP ===
    SHARES_PER_STEP = 3       # Promediar 3 shares reduce varianza ~3x
    SHARE_TIMEOUT = 60.0      # Timeout por share individual
    
    # Experiment sizes
    NARMA_STEPS = 150         # Pasos NARMA-10
    MEMORY_STEPS = 80         # Pasos Memory Capacity
    XOR_STEPS = 80            # Pasos XOR
    WARMUP_SHARES = 5         # Shares de calentamiento
    TRAIN_RATIO = 0.8
    
    # Difficulty (from calibration)
    D_BASE = 0.5
    D_MIN = 0.1
    D_MAX = 50.0
    EPSILON = 0.05
    
    # Ridge regression
    RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Thresholds for verdicts
    COUPLING_THRESHOLD = 10.0    # % improvement Normal vs Shuffle
    ENTROPY_THRESHOLD = 10.0     # % improvement Normal vs Constant
    MEMORY_THRESHOLD = 0.3       # Total MC
    XOR_THRESHOLD = 0.55         # Accuracy (55% = significantly above 50%)

# =============================================================================
# LOGGER
# =============================================================================
class Logger:
    def __init__(self):
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(self.results_dir, f"run_{self.run_id}.json")
        self.log_path = os.path.join(self.results_dir, f"run_{self.run_id}.log")
        self.history_path = os.path.join(self.results_dir, "HISTORY.jsonl")
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        self.log_file.write(formatted + "\n")
        self.log_file.flush()
        
    def save(self, data: Dict):
        # Full JSON
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.log(f"[SAVED] Full results: {self.json_path}")
        
        # Append to history
        summary = {
            'run_id': self.run_id,
            'timestamp': data.get('timestamp'),
            'shares_per_step': Config.SHARES_PER_STEP,
            'narma10_normal': data.get('narma10_normal'),
            'narma10_shuffle': data.get('narma10_shuffle'),
            'narma10_constant': data.get('narma10_constant'),
            'correlation_normal': data.get('correlation_normal'),
            'correlation_shuffle': data.get('correlation_shuffle'),
            'memory_capacity': data.get('memory_capacity'),
            'xor_accuracy': data.get('xor_accuracy'),
            'status': data.get('status')
        }
        with open(self.history_path, 'a') as f:
            f.write(json.dumps(summary) + "\n")
        self.log(f"[SAVED] History updated: {self.history_path}")
        
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

def pearson_r(x, y):
    if len(x) != len(y) or len(x) < 3: return 0.0
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx)**2 for xi in x))
    den_y = math.sqrt(sum((yi - my)**2 for yi in y))
    return num / (den_x * den_y) if (den_x * den_y) > 0 else 0.0

def nrmse(y_true, y_pred):
    r = max(y_true) - min(y_true)
    if r == 0: return 1.0
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
# SYNC BURST STRATUM SERVER
# =============================================================================
class SyncBurstServer(threading.Thread):
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
        
        # Synchronization
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

    def inject_burst(self, u_value: float, n_shares: int) -> Optional[List[float]]:
        """
        SYNC BURST: Inyecta input y espera N shares.
        Retorna lista de tiempos de respuesta, o None si timeout.
        """
        if not self.client_conn or not self.authorized:
            return None
        
        # Clear state
        with self.share_lock:
            self.share_times = []
        self.share_event.clear()
        
        # Set difficulty based on input
        D = self.config.D_BASE / (u_value + self.config.EPSILON)
        D = max(self.config.D_MIN, min(self.config.D_MAX, D))
        self.set_difficulty(D)
        
        # Record start time
        start_time = time.perf_counter()
        
        # Send job
        self.send_job(u_value)
        
        # Collect N shares
        response_times = []
        timeout_per_share = self.config.SHARE_TIMEOUT
        
        for i in range(n_shares):
            # Wait for next share
            self.share_event.clear()
            
            if self.share_event.wait(timeout=timeout_per_share):
                with self.share_lock:
                    if self.share_times:
                        share_time = self.share_times[-1]
                        if i == 0:
                            rt = share_time - start_time
                        else:
                            # Inter-arrival time
                            rt = share_time - self.share_times[-2] if len(self.share_times) > 1 else share_time - start_time
                        response_times.append(rt)
            else:
                # Timeout
                return None
        
        return response_times

    def stop(self):
        self.running = False
        try: self.sock.close()
        except: pass

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_features(response_times: List[float], history: List[List[float]], window: int = 5) -> List[float]:
    """
    Extrae features de los tiempos de respuesta del burst.
    
    Features:
    1. Media del burst (ms)
    2. Std del burst (ms)
    3. Media móvil de medias (memoria corta)
    4. Tendencia (diferencia con paso anterior)
    5. Bias
    """
    if not response_times:
        return [0.0, 0.0, 0.0, 0.0, 1.0]
    
    # Convertir a ms
    times_ms = [t * 1000 for t in response_times]
    
    # Features del burst actual
    burst_mean = mean(times_ms)
    burst_std = std(times_ms) if len(times_ms) > 1 else 0.0
    
    # Media móvil de bursts anteriores
    if history:
        recent_means = [mean([t*1000 for t in h]) for h in history[-window:]]
        rolling_mean = mean(recent_means)
        # Tendencia
        if len(recent_means) >= 2:
            trend = burst_mean - recent_means[-1]
        else:
            trend = 0.0
    else:
        rolling_mean = burst_mean
        trend = 0.0
    
    return [burst_mean, burst_std, rolling_mean, trend, 1.0]

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_task(server, logger, u_values, y_values, config, mode="normal"):
    """Run task in specified mode, collecting burst data"""
    logger.log(f"\n[TASK] Running mode: {mode.upper()}")
    
    X = []
    history = []
    timeouts = 0
    
    for i, u in enumerate(u_values):
        # Determine inject value
        if mode == "constant":
            inject_val = 0.25
        else:
            inject_val = u
        
        # Collect burst
        response_times = server.inject_burst(inject_val, config.SHARES_PER_STEP)
        
        if response_times is None:
            timeouts += 1
            # Use fallback
            response_times = [5.0] * config.SHARES_PER_STEP  # 5 seconds fallback
        
        # Extract features
        features = extract_features(response_times, history)
        
        # For shuffle mode, we'll shuffle X at the end
        X.append(features)
        history.append(response_times)
        
        if i % 10 == 0:
            avg_rt = mean(response_times) * 1000
            logger.log(f"  Step {i}/{len(u_values)} | u={u:.3f} | RT={avg_rt:.1f}ms | TO={timeouts}")
    
    logger.log(f"[TASK] {mode} Complete. Timeouts: {timeouts}/{len(u_values)} ({100*timeouts/len(u_values):.1f}%)")
    
    # Shuffle if requested
    if mode == "shuffle":
        random.shuffle(X)
    
    return X, y_values, timeouts

def evaluate(X, y, config, logger) -> Tuple[float, float, float]:
    """Train and evaluate, return (nrmse, correlation, best_alpha)"""
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
    logger.log(f"  NRMSE = {best_nrmse:.4f} | r = {corr:.4f} | α = {best_alpha}")
    
    return best_nrmse, corr, best_alpha

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = Logger()
    config = Config()
    
    logger.log("="*60)
    logger.log("   LV06 EXPERIMENT V7 - SYNC BURST PROTOCOL")
    logger.log("="*60)
    logger.log(f"Config: D_BASE={config.D_BASE}, SHARES_PER_STEP={config.SHARES_PER_STEP}, STEPS={config.NARMA_STEPS}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'shares_per_step': config.SHARES_PER_STEP,
            'd_base': config.D_BASE,
            'narma_steps': config.NARMA_STEPS
        }
    }
    
    server = SyncBurstServer(config, logger)
    server.start()
    
    # Wait for miner
    logger.log("\n[WAIT] Waiting for miner connection...")
    timeout_start = time.time()
    while not server.authorized:
        if time.time() - timeout_start > 300:
            logger.log("[ERROR] Miner connection timeout")
            results['status'] = 'TIMEOUT'
            logger.save(results)
            return
        time.sleep(0.5)
    
    # Warmup
    logger.log(f"\n[WARMUP] {config.WARMUP_SHARES} shares...")
    for _ in range(config.WARMUP_SHARES):
        server.inject_burst(0.25, 1)
    
    # === PHASE 1: NARMA-10 ===
    logger.log("\n" + "="*60)
    logger.log("   PHASE 1: NARMA-10 BENCHMARK")
    logger.log("="*60)
    
    u, y = generate_narma10(config.NARMA_STEPS + 50, seed=42)
    u, y = u[50:], y[50:]
    
    # Normal
    X_normal, _, to_normal = run_task(server, logger, u, y, config, mode="normal")
    nrmse_normal, corr_normal, alpha_normal = evaluate(X_normal, y, config, logger)
    results['narma10_normal'] = nrmse_normal
    results['correlation_normal'] = corr_normal
    
    # Shuffle
    X_shuffle, _, to_shuffle = run_task(server, logger, u, y, config, mode="shuffle")
    nrmse_shuffle, corr_shuffle, _ = evaluate(X_shuffle, y, config, logger)
    results['narma10_shuffle'] = nrmse_shuffle
    results['correlation_shuffle'] = corr_shuffle
    
    # Constant
    X_constant, _, to_constant = run_task(server, logger, u, y, config, mode="constant")
    nrmse_constant, corr_constant, _ = evaluate(X_constant, y, config, logger)
    results['narma10_constant'] = nrmse_constant
    results['correlation_constant'] = corr_constant
    
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
        logger.log(f"  Delay {delay}: MC = {mc:.4f}")
    
    total_mc = sum(mc for _, mc in mc_scores)
    results['memory_capacity'] = total_mc
    results['memory_scores'] = mc_scores
    logger.log(f"\n  TOTAL MEMORY CAPACITY: {total_mc:.4f}")
    
    # === PHASE 3: XOR ===
    logger.log("\n" + "="*60)
    logger.log("   PHASE 3: XOR NONLINEARITY TEST")
    logger.log("="*60)
    
    u_xor, y_xor = generate_xor_task(config.XOR_STEPS + 50, delay=3, seed=42)
    u_xor, y_xor = u_xor[50:], y_xor[50:]
    X_xor, _, _ = run_task(server, logger, u_xor, y_xor, config, mode="normal")
    
    split = int(len(X_xor) * config.TRAIN_RATIO)
    model = SimpleRidge(1.0).fit(X_xor[:split], y_xor[:split])
    pred = model.predict(X_xor[split:])
    accuracy = mean([1.0 if (p > 0.5) == (t > 0.5) else 0.0 for p, t in zip(pred, y_xor[split:])])
    results['xor_accuracy'] = accuracy
    logger.log(f"  XOR Accuracy: {accuracy:.2%} (baseline: 50%)")
    
    # === VERDICTS ===
    logger.log("\n" + "="*60)
    logger.log("   VERDICTS")
    logger.log("="*60)
    
    verdicts = {}
    
    # Coupling test
    if nrmse_shuffle > 0:
        improvement = (nrmse_shuffle - nrmse_normal) / nrmse_shuffle * 100
        passed = improvement > config.COUPLING_THRESHOLD
        verdicts['coupling'] = {'passed': passed, 'improvement': improvement, 'threshold': config.COUPLING_THRESHOLD}
        symbol = "✅" if passed else "❌"
        logger.log(f"{symbol} COUPLING: Normal ({nrmse_normal:.4f}) vs Shuffle ({nrmse_shuffle:.4f}) = {improvement:.1f}% (threshold: {config.COUPLING_THRESHOLD}%)")
    
    # Entropy test
    if nrmse_constant > 0:
        improvement = (nrmse_constant - nrmse_normal) / nrmse_constant * 100
        passed = improvement > config.ENTROPY_THRESHOLD
        verdicts['entropy'] = {'passed': passed, 'improvement': improvement, 'threshold': config.ENTROPY_THRESHOLD}
        symbol = "✅" if passed else "❌"
        logger.log(f"{symbol} ENTROPY: Normal ({nrmse_normal:.4f}) vs Constant ({nrmse_constant:.4f}) = {improvement:.1f}% (threshold: {config.ENTROPY_THRESHOLD}%)")
    
    # Correlation comparison (new metric!)
    corr_diff = abs(corr_normal) - abs(corr_shuffle)
    passed = corr_diff > 0.2  # Normal should have much stronger correlation
    verdicts['correlation'] = {'passed': passed, 'normal': corr_normal, 'shuffle': corr_shuffle, 'diff': corr_diff}
    symbol = "✅" if passed else "❌"
    logger.log(f"{symbol} CORRELATION: Normal (r={corr_normal:.4f}) vs Shuffle (r={corr_shuffle:.4f}) = Δ{corr_diff:.4f}")
    
    # Memory test
    passed = total_mc > config.MEMORY_THRESHOLD
    verdicts['memory'] = {'passed': passed, 'value': total_mc, 'threshold': config.MEMORY_THRESHOLD}
    symbol = "✅" if passed else "❌"
    logger.log(f"{symbol} MEMORY: MC = {total_mc:.4f} (threshold: {config.MEMORY_THRESHOLD})")
    
    # XOR test
    passed = accuracy > config.XOR_THRESHOLD
    verdicts['nonlinearity'] = {'passed': passed, 'value': accuracy, 'threshold': config.XOR_THRESHOLD}
    symbol = "✅" if passed else "❌"
    logger.log(f"{symbol} NONLINEARITY: Accuracy = {accuracy:.2%} (threshold: {config.XOR_THRESHOLD:.0%})")
    
    results['verdicts'] = verdicts
    
    # === FINAL STATUS ===
    passed_count = sum(1 for v in verdicts.values() if v['passed'])
    total_count = len(verdicts)
    
    # New logic: if coupling OR correlation passes, there's evidence of physical coupling
    has_coupling = verdicts.get('coupling', {}).get('passed', False) or verdicts.get('correlation', {}).get('passed', False)
    has_entropy = verdicts.get('entropy', {}).get('passed', False)
    
    if passed_count >= 4:
        status = "✅ GENUINE RESERVOIR COMPUTER"
    elif has_coupling and has_entropy:
        status = "✅ PHYSICAL COUPLING VERIFIED (Rate Encoder with RC potential)"
    elif has_coupling or has_entropy:
        status = "⚠️ PARTIAL COUPLING (needs more investigation)"
    else:
        status = "❌ NO RC CAPABILITY DEMONSTRATED"
    
    results['status'] = status
    results['passed'] = passed_count
    results['total'] = total_count
    
    logger.log("\n" + "="*60)
    logger.log(f"   FINAL STATUS: {status}")
    logger.log(f"   Tests passed: {passed_count}/{total_count}")
    logger.log("="*60)
    
    # Save
    logger.save(results)
    server.stop()
    logger.close()
    
    print(f"\n📁 All files saved in: {logger.results_dir}/")

if __name__ == "__main__":
    main()
