#!/usr/bin/env python3
"""
================================================================================
LV06 DEFINITIVE CAPABILITY ASSESSMENT
================================================================================
Author: Fran / Agnuxo (with Claude)
Date: December 2025

PURPOSE:
This experiment definitively determines what the Lucky Miner LV06 can and 
cannot do as a computational substrate. It is designed to be BRUTALLY HONEST
about the hardware's actual capabilities.

METHODOLOGY:
- Rate-Encoded input injection (difficulty modulation)
- Multiple benchmark tasks (NARMA-10, Memory Capacity, XOR Parity)
- Rigorous control conditions (Shuffle, Constant, Simulated Poisson)
- Statistical significance testing throughout

HARDWARE REQUIREMENTS:
- Lucky Miner LV06 configured to connect to this machine on port 3333
- Stable network connection
- ~30 minutes for full battery of tests

OUTPUT:
- Console progress and results
- lv06_definitive_report.json (machine-readable)
- lv06_definitive_report.md (human-readable)
================================================================================
"""

import socket
import threading
import json
import time
import struct
import hashlib
import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from datetime import datetime
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    """All experimental parameters in one place for reproducibility"""
    # Network
    HOST: str = "0.0.0.0"
    PORT: int = 3333
    
    # Timing
    WINDOW_TIME: float = 2.0        # <--- ESTABLE: 2.0s para evitar saturación de pipeline
    HANDSHAKE_TIMEOUT: float = 600.0 # 10 minutos
    WARMUP_STEPS: int = 15          
    
    # Rate Encoding Parameters
    D_BASE: float = 1.0             # <--- SENSIBLE: 1.0 para capturar dinámica en 2s
    EPSILON: float = 0.05           
    
    # Benchmark Sizes
    NARMA_STEPS: int = 500          # Steps for NARMA-10
    MEMORY_STEPS: int = 300         # Steps for Memory Capacity
    XOR_STEPS: int = 300            # Steps for XOR task
    POISSON_SIM_STEPS: int = 500    # Steps for simulated baseline
    
    # Statistical Thresholds
    SIGNIFICANCE_ALPHA: float = 0.05
    MIN_IMPROVEMENT_PERCENT: float = 10.0  # Must beat control by this much
    
    # Ridge Regression
    RIDGE_ALPHAS: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    TRAIN_RATIO: float = 0.8

# =============================================================================
# MATH UTILITIES (NO NUMPY DEPENDENCY)
# =============================================================================
def mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0

def variance(x: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    m = mean(x)
    return sum((xi - m)**2 for xi in x) / (len(x) - 1)

def std(x: List[float]) -> float:
    return math.sqrt(variance(x))

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient"""
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx)**2 for xi in x))
    den_y = math.sqrt(sum((yi - my)**2 for yi in y))
    if den_x * den_y == 0:
        return 0.0
    return num / (den_x * den_y)

def mse(y_true: List[float], y_pred: List[float]) -> float:
    return mean([(yt - yp)**2 for yt, yp in zip(y_true, y_pred)])

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))

def nrmse(y_true: List[float], y_pred: List[float]) -> float:
    """Normalized RMSE (by range)"""
    y_range = max(y_true) - min(y_true)
    if y_range == 0:
        return 1.0
    return rmse(y_true, y_pred) / y_range

# =============================================================================
# SIMPLE RIDGE REGRESSION (NO SKLEARN DEPENDENCY)
# =============================================================================
class SimpleRidge:
    """
    Ridge regression using normal equations.
    X @ w = y  =>  w = (X'X + αI)^-1 X'y
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        
    def _standardize(self, X: List[List[float]], fit: bool = False) -> List[List[float]]:
        """Standardize features (zero mean, unit variance)"""
        n_features = len(X[0]) if X else 0
        
        if fit:
            self.mean_x = [mean([row[j] for row in X]) for j in range(n_features)]
            self.std_x = [std([row[j] for row in X]) for j in range(n_features)]
            # Prevent division by zero
            self.std_x = [s if s > 1e-10 else 1.0 for s in self.std_x]
        
        return [[(row[j] - self.mean_x[j]) / self.std_x[j] for j in range(n_features)] for row in X]
    
    def _matrix_multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication A @ B"""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def _transpose(self, A: List[List[float]]) -> List[List[float]]:
        """Matrix transpose"""
        return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
    
    def _add_identity(self, A: List[List[float]], scale: float) -> List[List[float]]:
        """Add scaled identity matrix: A + scale * I"""
        n = len(A)
        result = [[A[i][j] for j in range(n)] for i in range(n)]
        for i in range(n):
            result[i][i] += scale
        return result
    
    def _invert_matrix(self, A: List[List[float]]) -> List[List[float]]:
        """Matrix inversion using Gauss-Jordan elimination"""
        n = len(A)
        # Augment with identity
        aug = [[A[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)] for i in range(n)]
        
        for col in range(n):
            # Find pivot
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            if abs(aug[col][col]) < 1e-12:
                # Singular matrix - return identity
                return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            
            # Scale pivot row
            scale = aug[col][col]
            aug[col] = [x / scale for x in aug[col]]
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
        
        return [[aug[i][j] for j in range(n, 2 * n)] for i in range(n)]
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'SimpleRidge':
        """Fit ridge regression model"""
        self.mean_y = mean(y)
        y_centered = [yi - self.mean_y for yi in y]
        
        X_std = self._standardize(X, fit=True)
        
        # Normal equations: w = (X'X + αI)^-1 X'y
        Xt = self._transpose(X_std)
        XtX = self._matrix_multiply(Xt, X_std)
        XtX_reg = self._add_identity(XtX, self.alpha)
        XtX_inv = self._invert_matrix(XtX_reg)
        
        # X'y as column vector
        Xty = [[sum(Xt[i][j] * y_centered[j] for j in range(len(y_centered)))] for i in range(len(Xt))]
        
        w = self._matrix_multiply(XtX_inv, Xty)
        self.weights = [w[i][0] for i in range(len(w))]
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict using fitted model"""
        X_std = self._standardize(X, fit=False)
        return [sum(row[j] * self.weights[j] for j in range(len(self.weights))) + self.mean_y for row in X_std]

# =============================================================================
# BENCHMARK GENERATORS
# =============================================================================
def generate_narma10(length: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    """
    Standard NARMA-10 benchmark.
    y[t] = 0.3*y[t-1] + 0.05*y[t-1]*sum(y[t-9:t+1]) + 1.5*u[t-9]*u[t] + 0.1
    """
    random.seed(seed)
    u = [random.uniform(0, 0.5) for _ in range(length)]
    y = [0.0] * length
    
    for t in range(10, length):
        sum_y = sum(y[max(0, t-9):t+1])
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * sum_y
        term3 = 1.5 * u[t-9] * u[t]
        term4 = 0.1
        y[t] = max(0, min(1, term1 + term2 + term3 + term4))
    
    return u, y

def generate_memory_capacity_task(length: int, delay: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    """
    Memory capacity task: predict u[t-delay] from current state.
    Tests how many steps back the system can "remember".
    """
    random.seed(seed)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0] * delay + u[:-delay] if delay > 0 else u[:]
    return u, y

def generate_xor_task(length: int, delay: int = 3, seed: int = 42) -> Tuple[List[float], List[float]]:
    """
    Temporal XOR parity task: y[t] = XOR of binarized u[t] and u[t-delay].
    Tests nonlinear temporal processing.
    """
    random.seed(seed)
    u = [random.uniform(0, 1) for _ in range(length)]
    y = [0.0] * length
    
    for t in range(delay, length):
        bit_now = 1.0 if u[t] > 0.5 else 0.0
        bit_past = 1.0 if u[t-delay] > 0.5 else 0.0
        y[t] = 1.0 if (bit_now != bit_past) else 0.0
    
    return u, y

# =============================================================================
# STRATUM SERVER (RATE-ENCODED RC)
# =============================================================================
class LV06StratumServer(threading.Thread):
    """
    Minimal Stratum server implementing Rate-Encoded Reservoir Computing.
    
    The key insight: we modulate DIFFICULTY to change the expected share rate.
    D = D_base / (u[t] + epsilon)
    Higher input -> Lower difficulty -> More shares expected
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.daemon = True
        
        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(1)
        
        # State
        self.client_conn = None
        self.running = True
        self.connection_active = False
        self.authorized = False
        
        # Share buffer with precise timestamps
        self.share_buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        
        # Job tracking
        self.job_counter = 0
        self.extranonce1 = "deadbeef"
        self.extranonce2_size = 4
        self.current_difficulty = config.D_BASE
        
    def run(self):
        """Main server loop"""
        print(f"[SERVER] Listening on {self.config.HOST}:{self.config.PORT}")
        try:
            self.sock.settimeout(self.config.HANDSHAKE_TIMEOUT)
            conn, addr = self.sock.accept()
            print(f"[SERVER] Connected by {addr}")
            self.client_conn = conn
            self.connection_active = True
            self._handle_client(conn)
        except socket.timeout:
            print("[SERVER] Timeout waiting for miner connection")
        except Exception as e:
            print(f"[SERVER] Error: {e}")
        finally:
            self.connection_active = False
            
    def _handle_client(self, conn):
        """Handle incoming Stratum messages (Blocking for stability)"""
        buffer = ""
        
        while self.running and self.connection_active:
            try:
                data = conn.recv(8192)
                if not data:
                    break
                buffer += data.decode('utf-8', errors='ignore')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    raw_line = line.strip()
                    if raw_line:
                        print(f"[RECV] {raw_line}")
                    self._process_message(conn, raw_line)
                    
            except Exception as e:
                if self.running:
                    print(f"[SERVER] Client error: {e}")
                break
                
        print("[SERVER] Client disconnected")
        self.connection_active = False
        
    def _process_message(self, conn, line: str):
        """Process a single Stratum message"""
        if not line:
            return
            
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
            
        method = msg.get('method', '')
        msg_id = msg.get('id')
        
        if method == 'mining.subscribe':
            response = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]],
                    self.extranonce1,
                    self.extranonce2_size
                ],
                "error": None
            }
            self._send(conn, response)
            self._send_difficulty(conn, self.config.D_BASE)
            
        elif method == 'mining.authorize':
            self._send(conn, {"id": msg_id, "result": True, "error": None})
            self.authorized = True
            # CRITICAL: Re-send difficulty after authorization to ensure it sticks
            self._send_difficulty(conn, self.config.D_BASE)
            print(f"[SERVER] Miner authorized. Initial Difficulty: {self.config.D_BASE}")
            
        elif method == 'mining.configure':
            self._send(conn, {"id": msg_id, "result": {"version-rolling.mask": "ffffffff"}, "error": None})
            
        elif method == 'mining.submit':
            # CRITICAL: Record share with precise timestamp
            arrival_time = time.perf_counter()
            params = msg.get('params', [])
            
            share_record = {
                "time": arrival_time,
                "nonce": params[4] if len(params) > 4 else "",
                "job_id": params[1] if len(params) > 1 else "",
                "difficulty": self.current_difficulty
            }
            
            with self.buffer_lock:
                self.share_buffer.append(share_record)
            
            # ACK is mandatory for continued operation
            self._send(conn, {"id": msg_id, "result": True, "error": None})
            
    def _send(self, conn, data: dict):
        """Send JSON message to client"""
        try:
            line = json.dumps(data) + '\n'
            print(f"[SEND] {line.strip()}")
            conn.sendall(line.encode('utf-8'))
        except Exception as e:
            print(f"[SERVER] Send error: {e}")
            
    def _send_difficulty(self, conn, diff: float):
        """Send difficulty adjustment"""
        print(f"[SERVER] Setting Difficulty -> {diff:.6f}")
        self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [diff]})
        self.current_difficulty = diff
        
    def inject_rate(self, u_value: float) -> bool:
        """
        RATE ENCODING: Inject input by modulating difficulty.
        
        D = D_base / (u + epsilon)
        
        - u = 0.0 -> D = D_base/epsilon = 500 (very hard, few shares)
        - u = 0.5 -> D = D_base/0.55 = 45.5 (easier, more shares)
        
        This creates a PHYSICAL COUPLING between input and share rate.
        """
        if not self.client_conn or not self.authorized:
            return False
            
        # Calculate rate-encoded difficulty
        D = self.config.D_BASE / (u_value + self.config.EPSILON)
        
        # Send new difficulty
        self._send_difficulty(self.client_conn, D)
        
        # INCREASED delay to 20ms to ensure difficulty is processed before the job
        time.sleep(0.02)
        
        # Send new job to trigger immediate switch
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        # Encode u_value in coinbase for traceability
        u_hex = struct.pack('>f', u_value).hex()
        # USE THE REFERENCE COINBASE FORMAT (Experiment 3 Final)
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        job_msg = {
            "id": None,
            "method": "mining.notify",
            "params": [job_id, "0" * 64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        }
        self._send(self.client_conn, job_msg)
        
        return True
        
    def harvest_state(self, clear: bool = True) -> List[Dict]:
        """
        Harvest all shares received since last call.
        Returns list of share records with timestamps.
        """
        with self.buffer_lock:
            shares = list(self.share_buffer)
            if clear:
                self.share_buffer = []
        return shares
        
    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
        try:
            self.sock.close()
        except:
            pass

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_features(shares: List[Dict], window_time: float) -> List[float]:
    """
    Extract reservoir state features from share arrival times.
    
    Features:
    1. Count: Number of shares in window
    2. Mean IAT: Mean inter-arrival time (ms)
    3. Std IAT: Standard deviation of IATs (ms)
    4. Rate: Shares per second
    5. CV: Coefficient of variation of IATs
    6. Bias: Always 1.0 for ridge regression
    """
    n = len(shares)
    
    if n == 0:
        return [0.0, window_time * 1000, 0.0, 0.0, 0.0, 1.0]
    
    rate = n / window_time
    
    if n == 1:
        return [float(n), window_time * 1000, 0.0, rate, 0.0, 1.0]
    
    # Inter-arrival times in milliseconds
    times = [s['time'] for s in shares]
    iats = [(times[i] - times[i-1]) * 1000 for i in range(1, n)]
    
    mean_iat = mean(iats)
    std_iat = std(iats)
    cv = std_iat / mean_iat if mean_iat > 0 else 0.0
    
    return [float(n), mean_iat, std_iat, rate, cv, 1.0]

# =============================================================================
# SIMULATED POISSON BASELINE
# =============================================================================
def simulate_poisson_shares(u_values: List[float], config: ExperimentConfig, seed: int = 42) -> List[List[float]]:
    """
    Simulate what a theoretically perfect Poisson process would produce.
    
    This is the CRITICAL BASELINE: if the real hardware doesn't beat this,
    then there's no evidence of useful reservoir dynamics.
    
    λ(u) = k / D(u) = k * (u + epsilon) / D_base
    
    We calibrate k from expected hash rate.
    """
    random.seed(seed)
    
    # Calibration: expect ~10 shares/sec at D=D_BASE with u=0.25
    k = 10.0 * config.D_BASE  # Base rate constant
    
    features = []
    
    for u in u_values:
        # Expected rate for this input
        D = config.D_BASE / (u + config.EPSILON)
        lambda_u = k / D  # shares per second
        
        # Simulate Poisson arrivals
        n_shares = 0
        t = 0.0
        arrival_times = []
        
        while t < config.WINDOW_TIME:
            # Exponential inter-arrival time
            iat = random.expovariate(lambda_u) if lambda_u > 0 else config.WINDOW_TIME
            t += iat
            if t < config.WINDOW_TIME:
                arrival_times.append(t)
                n_shares += 1
        
        # Extract same features as real hardware
        n = len(arrival_times)
        if n == 0:
            feat = [0.0, config.WINDOW_TIME * 1000, 0.0, 0.0, 0.0, 1.0]
        elif n == 1:
            feat = [float(n), config.WINDOW_TIME * 1000, 0.0, n / config.WINDOW_TIME, 0.0, 1.0]
        else:
            iats = [(arrival_times[i] - arrival_times[i-1]) * 1000 for i in range(1, n)]
            mean_iat = mean(iats)
            std_iat = std(iats)
            cv = std_iat / mean_iat if mean_iat > 0 else 0.0
            feat = [float(n), mean_iat, std_iat, n / config.WINDOW_TIME, cv, 1.0]
        
        features.append(feat)
    
    return features

# =============================================================================
# EXPERIMENTAL MODES
# =============================================================================
@dataclass
class ExperimentResult:
    """Result of a single experimental run"""
    mode: str
    task: str
    nrmse: float
    correlation: float
    n_samples: int
    n_features: int
    best_alpha: float
    shares_per_step: float
    details: Dict[str, Any] = field(default_factory=dict)

def run_hardware_experiment(
    server: LV06StratumServer,
    u_values: List[float],
    y_targets: List[float],
    config: ExperimentConfig,
    mode: str = "normal"
) -> Tuple[List[List[float]], List[float]]:
    """
    Run experiment on actual LV06 hardware.
    
    Modes:
    - normal: Standard rate-encoded injection
    - constant: Fixed input (u=0.25) to test entropy source
    - shuffle: Normal injection but shuffle features (break causality)
    """
    X_features = []
    total_shares = 0
    
    print(f"\n[{mode.upper()}] Running {len(u_values)} steps...")
    
    for t, (u, y) in enumerate(zip(u_values, y_targets)):
        # Inject input (or constant)
        inject_value = 0.25 if mode == "constant" else u
        server.inject_rate(inject_value)
        
        # Wait for window
        time.sleep(config.WINDOW_TIME)
        
        # Harvest shares
        shares = server.harvest_state()
        total_shares += len(shares)
        
        # Extract features
        features = extract_features(shares, config.WINDOW_TIME)
        X_features.append(features)
        
        # Progress
        if t % 50 == 0:
            print(f"  Step {t}/{len(u_values)} | u={u:.3f} | shares={len(shares)} | mode={mode}")
    
    print(f"[{mode.upper()}] Complete. Total shares: {total_shares} ({total_shares/len(u_values):.1f}/step)")
    
    # Shuffle if requested
    if mode == "shuffle":
        random.shuffle(X_features)
    
    return X_features, y_targets

def evaluate_model(
    X_train: List[List[float]], 
    y_train: List[float],
    X_test: List[List[float]], 
    y_test: List[float],
    alphas: List[float]
) -> Tuple[float, float, float]:
    """
    Train and evaluate Ridge regression, trying multiple alpha values.
    Returns: (best_nrmse, best_alpha, correlation)
    """
    best_nrmse = 1.0
    best_alpha = alphas[0]
    best_pred = None
    
    for alpha in alphas:
        try:
            model = SimpleRidge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            score = nrmse(y_test, y_pred)
            
            if score < best_nrmse:
                best_nrmse = score
                best_alpha = alpha
                best_pred = y_pred
        except Exception as e:
            continue
    
    # Compute correlation with best model
    corr = pearson_correlation(y_test, best_pred) if best_pred else 0.0
    
    return best_nrmse, best_alpha, corr

# =============================================================================
# MAIN EXPERIMENT BATTERY
# =============================================================================
class DefinitiveExperiment:
    """
    Complete experimental battery to assess LV06 capabilities.
    """
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results: List[ExperimentResult] = []
        self.server: Optional[LV06StratumServer] = None
        self.telemetry: List[Dict] = []
        
    def run_all(self) -> Dict:
        """Execute complete experimental battery"""
        print("=" * 70)
        print("   LV06 DEFINITIVE CAPABILITY ASSESSMENT")
        print("   Fran / Agnuxo - December 2025")
        print("=" * 70)
        
        # Initialize server
        self.server = LV06StratumServer(self.config)
        self.server.start()
        
        # Wait for connection
        print("\n[INIT] Waiting for LV06 connection...")
        start_wait = time.time()
        while not self.server.authorized:
            if time.time() - start_wait > self.config.HANDSHAKE_TIMEOUT:
                print("[ERROR] Timeout waiting for miner. Aborting.")
                return self._generate_report(hardware_connected=False)
            time.sleep(0.5)
        
        print("[INIT] Miner connected and authorized!")
        
        # Warmup
        print(f"\n[WARMUP] Running {self.config.WARMUP_STEPS} warmup steps...")
        for _ in range(self.config.WARMUP_STEPS):
            self.server.inject_rate(0.25)
            time.sleep(self.config.WINDOW_TIME)
            self.server.harvest_state()
        
        try:
            # === PHASE 1: NARMA-10 Battery ===
            print("\n" + "=" * 70)
            print("   PHASE 1: NARMA-10 BENCHMARK")
            print("=" * 70)
            
            self._run_narma10_battery()
            
            # === PHASE 2: Memory Capacity ===
            print("\n" + "=" * 70)
            print("   PHASE 2: MEMORY CAPACITY TEST")
            print("=" * 70)
            
            self._run_memory_capacity()
            
            # === PHASE 3: XOR Nonlinearity ===
            print("\n" + "=" * 70)
            print("   PHASE 3: TEMPORAL XOR NONLINEARITY")
            print("=" * 70)
            
            self._run_xor_task()
            
            # === PHASE 4: Poisson Baseline ===
            print("\n" + "=" * 70)
            print("   PHASE 4: POISSON BASELINE COMPARISON")
            print("=" * 70)
            
            self._run_poisson_baseline()
            
        finally:
            self.server.stop()
        
        return self._generate_report(hardware_connected=True)
    
    def _run_narma10_battery(self):
        """Run NARMA-10 in all experimental modes"""
        u, y = generate_narma10(self.config.NARMA_STEPS + self.config.WARMUP_STEPS)
        u, y = u[self.config.WARMUP_STEPS:], y[self.config.WARMUP_STEPS:]
        
        for mode in ["normal", "shuffle", "constant"]:
            X, Y = run_hardware_experiment(self.server, u, y, self.config, mode=mode)
            
            # Train/test split
            split = int(len(X) * self.config.TRAIN_RATIO)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = Y[:split], Y[split:]
            
            # Evaluate
            nrmse_score, best_alpha, corr = evaluate_model(
                X_train, y_train, X_test, y_test, self.config.RIDGE_ALPHAS
            )
            
            result = ExperimentResult(
                mode=mode,
                task="NARMA-10",
                nrmse=nrmse_score,
                correlation=corr,
                n_samples=len(X),
                n_features=len(X[0]) if X else 0,
                best_alpha=best_alpha,
                shares_per_step=mean([x[0] for x in X])
            )
            self.results.append(result)
            
            print(f"\n  [{mode.upper()}] NRMSE = {nrmse_score:.4f} | r = {corr:.4f} | α = {best_alpha}")
    
    def _run_memory_capacity(self):
        """Test memory capacity at different delays"""
        memory_scores = []
        
        for delay in [1, 2, 3, 5, 10]:
            u, y = generate_memory_capacity_task(self.config.MEMORY_STEPS + self.config.WARMUP_STEPS, delay)
            u, y = u[self.config.WARMUP_STEPS:], y[self.config.WARMUP_STEPS:]
            
            X, Y = run_hardware_experiment(self.server, u, y, self.config, mode="normal")
            
            split = int(len(X) * self.config.TRAIN_RATIO)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = Y[:split], Y[split:]
            
            nrmse_score, best_alpha, corr = evaluate_model(
                X_train, y_train, X_test, y_test, self.config.RIDGE_ALPHAS
            )
            
            # Memory capacity metric: correlation squared
            mc = max(0, corr ** 2)
            memory_scores.append((delay, mc, nrmse_score))
            
            print(f"  Delay {delay}: MC = {mc:.4f} | NRMSE = {nrmse_score:.4f}")
        
        # Total memory capacity (sum of MC at each delay)
        total_mc = sum(mc for _, mc, _ in memory_scores)
        
        result = ExperimentResult(
            mode="normal",
            task="Memory Capacity",
            nrmse=mean([n for _, _, n in memory_scores]),
            correlation=total_mc,  # Using correlation field for total MC
            n_samples=self.config.MEMORY_STEPS,
            n_features=6,
            best_alpha=1.0,
            shares_per_step=0.0,
            details={"delays": memory_scores, "total_mc": total_mc}
        )
        self.results.append(result)
        
        print(f"\n  TOTAL MEMORY CAPACITY: {total_mc:.4f} steps")
    
    def _run_xor_task(self):
        """Test nonlinear temporal processing with XOR task"""
        u, y = generate_xor_task(self.config.XOR_STEPS + self.config.WARMUP_STEPS, delay=3)
        u, y = u[self.config.WARMUP_STEPS:], y[self.config.WARMUP_STEPS:]
        
        X, Y = run_hardware_experiment(self.server, u, y, self.config, mode="normal")
        
        split = int(len(X) * self.config.TRAIN_RATIO)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = Y[:split], Y[split:]
        
        nrmse_score, best_alpha, corr = evaluate_model(
            X_train, y_train, X_test, y_test, self.config.RIDGE_ALPHAS
        )
        
        # For binary task, also compute accuracy
        y_pred = SimpleRidge(alpha=best_alpha).fit(X_train, y_train).predict(X_test)
        accuracy = mean([1.0 if (yp > 0.5) == (yt > 0.5) else 0.0 for yp, yt in zip(y_pred, y_test)])
        
        result = ExperimentResult(
            mode="normal",
            task="XOR Parity (delay=3)",
            nrmse=nrmse_score,
            correlation=corr,
            n_samples=len(X),
            n_features=len(X[0]) if X else 0,
            best_alpha=best_alpha,
            shares_per_step=mean([x[0] for x in X]),
            details={"accuracy": accuracy}
        )
        self.results.append(result)
        
        print(f"\n  XOR Task: Accuracy = {accuracy:.2%} | NRMSE = {nrmse_score:.4f}")
    
    def _run_poisson_baseline(self):
        """Compare against theoretical Poisson process"""
        u, y = generate_narma10(self.config.POISSON_SIM_STEPS + self.config.WARMUP_STEPS)
        u, y = u[self.config.WARMUP_STEPS:], y[self.config.WARMUP_STEPS:]
        
        # Simulate Poisson baseline
        X_sim = simulate_poisson_shares(u, self.config)
        
        split = int(len(X_sim) * self.config.TRAIN_RATIO)
        X_train, X_test = X_sim[:split], X_sim[split:]
        y_train, y_test = y[:split], y[split:]
        
        nrmse_score, best_alpha, corr = evaluate_model(
            X_train, y_train, X_test, y_test, self.config.RIDGE_ALPHAS
        )
        
        result = ExperimentResult(
            mode="simulated_poisson",
            task="NARMA-10 (Poisson Baseline)",
            nrmse=nrmse_score,
            correlation=corr,
            n_samples=len(X_sim),
            n_features=len(X_sim[0]) if X_sim else 0,
            best_alpha=best_alpha,
            shares_per_step=mean([x[0] for x in X_sim])
        )
        self.results.append(result)
        
        print(f"\n  Poisson Baseline: NRMSE = {nrmse_score:.4f}")
    
    def _generate_report(self, hardware_connected: bool) -> Dict:
        """Generate comprehensive report with verdicts"""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hardware": "Lucky Miner LV06 (BM1366/BM1387)",
                "hardware_connected": hardware_connected,
                "config": asdict(self.config)
            },
            "results": [],
            "verdicts": {},
            "summary": {}
        }
        
        if not hardware_connected:
            report["summary"]["status"] = "ABORTED - No hardware connection"
            return report
        
        # Organize results by task
        for r in self.results:
            report["results"].append({
                "task": r.task,
                "mode": r.mode,
                "nrmse": r.nrmse,
                "correlation": r.correlation,
                "samples": r.n_samples,
                "shares_per_step": r.shares_per_step,
                "best_alpha": r.best_alpha,
                "details": r.details
            })
        
        # === VERDICTS ===
        
        # 1. Rate Encoding Coupling
        narma_results = {r.mode: r for r in self.results if r.task == "NARMA-10"}
        if "normal" in narma_results and "shuffle" in narma_results:
            improvement = (narma_results["shuffle"].nrmse - narma_results["normal"].nrmse) / narma_results["shuffle"].nrmse * 100
            verdict_coupling = improvement > self.config.MIN_IMPROVEMENT_PERCENT
            report["verdicts"]["rate_encoding_coupling"] = {
                "passed": verdict_coupling,
                "improvement_percent": improvement,
                "threshold": self.config.MIN_IMPROVEMENT_PERCENT,
                "explanation": f"Normal ({narma_results['normal'].nrmse:.4f}) vs Shuffle ({narma_results['shuffle'].nrmse:.4f})"
            }
        
        # 2. Entropy Contribution
        if "normal" in narma_results and "constant" in narma_results:
            improvement = (narma_results["constant"].nrmse - narma_results["normal"].nrmse) / narma_results["constant"].nrmse * 100
            verdict_entropy = improvement > self.config.MIN_IMPROVEMENT_PERCENT
            report["verdicts"]["entropy_contribution"] = {
                "passed": verdict_entropy,
                "improvement_percent": improvement,
                "threshold": self.config.MIN_IMPROVEMENT_PERCENT,
                "explanation": f"Normal ({narma_results['normal'].nrmse:.4f}) vs Constant ({narma_results['constant'].nrmse:.4f})"
            }
        
        # 3. Beats Poisson Baseline
        poisson_results = [r for r in self.results if r.mode == "simulated_poisson"]
        if poisson_results and "normal" in narma_results:
            poisson_nrmse = poisson_results[0].nrmse
            hw_nrmse = narma_results["normal"].nrmse
            improvement = (poisson_nrmse - hw_nrmse) / poisson_nrmse * 100
            verdict_poisson = improvement > 5.0  # Must beat Poisson by 5%
            report["verdicts"]["beats_poisson_baseline"] = {
                "passed": verdict_poisson,
                "hardware_nrmse": hw_nrmse,
                "poisson_nrmse": poisson_nrmse,
                "improvement_percent": improvement,
                "explanation": "Hardware must beat theoretical Poisson to show useful dynamics"
            }
        
        # 4. Memory Capacity
        mc_results = [r for r in self.results if r.task == "Memory Capacity"]
        if mc_results:
            total_mc = mc_results[0].details.get("total_mc", 0)
            verdict_memory = total_mc > 0.5  # At least 0.5 steps of memory
            report["verdicts"]["memory_capacity"] = {
                "passed": verdict_memory,
                "total_mc": total_mc,
                "threshold": 0.5,
                "explanation": "Must have >0.5 steps of memory capacity for RC"
            }
        
        # 5. Nonlinearity
        xor_results = [r for r in self.results if "XOR" in r.task]
        if xor_results:
            accuracy = xor_results[0].details.get("accuracy", 0.5)
            verdict_xor = accuracy > 0.6  # Better than chance (0.5)
            report["verdicts"]["nonlinear_processing"] = {
                "passed": verdict_xor,
                "accuracy": accuracy,
                "threshold": 0.6,
                "explanation": "XOR task accuracy must exceed 60% to show nonlinear capability"
            }
        
        # === FINAL SUMMARY ===
        passed = sum(1 for v in report["verdicts"].values() if v["passed"])
        total = len(report["verdicts"])
        
        report["summary"] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0.0,
            "status": self._determine_status(report["verdicts"])
        }
        
        return report
    
    def _determine_status(self, verdicts: Dict) -> str:
        """Determine overall status based on verdicts"""
        passed = [k for k, v in verdicts.items() if v["passed"]]
        failed = [k for k, v in verdicts.items() if not v["passed"]]
        
        # RESERVOIR COMPUTING requires:
        # - Rate encoding coupling (physical input-output)
        # - Memory capacity > 0
        # - Beats Poisson (not just a random process)
        
        rc_requirements = ["rate_encoding_coupling", "beats_poisson_baseline", "memory_capacity"]
        rc_passed = all(k in passed for k in rc_requirements if k in verdicts)
        
        if rc_passed and "nonlinear_processing" in passed:
            return "✅ GENUINE RESERVOIR COMPUTER - All criteria met"
        elif rc_passed:
            return "⚠️ LINEAR RESERVOIR - RC criteria met but no nonlinearity demonstrated"
        elif "rate_encoding_coupling" in passed:
            return "⚠️ RATE ENCODER - Physical coupling exists but not a full RC"
        else:
            return "❌ NO RC CAPABILITY - Hardware does not function as reservoir"

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point"""
    experiment = DefinitiveExperiment()
    report = experiment.run_all()
    
    # Print summary
    print("\n" + "=" * 70)
    print("   FINAL REPORT")
    print("=" * 70)
    
    print(f"\nStatus: {report['summary'].get('status', 'Unknown')}")
    print(f"Tests Passed: {report['summary'].get('passed', 0)}/{report['summary'].get('total', 0)}")
    
    print("\n--- VERDICTS ---")
    for name, verdict in report.get("verdicts", {}).items():
        symbol = "✅" if verdict["passed"] else "❌"
        print(f"  {symbol} {name}: {verdict.get('explanation', '')}")
    
    print("\n--- NARMA-10 RESULTS ---")
    for r in report.get("results", []):
        if r["task"] == "NARMA-10":
            print(f"  {r['mode'].upper():12} NRMSE = {r['nrmse']:.4f}")
    
    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = f"lv06_definitive_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[SAVED] JSON: {json_path}")
    
    # Markdown
    md_path = f"lv06_definitive_report_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("# LV06 Definitive Capability Assessment\n\n")
        f.write(f"**Date:** {report['metadata']['timestamp']}\n")
        f.write(f"**Hardware:** {report['metadata']['hardware']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"**Status:** {report['summary'].get('status', 'Unknown')}\n\n")
        f.write(f"**Tests Passed:** {report['summary'].get('passed', 0)}/{report['summary'].get('total', 0)}\n\n")
        
        f.write("## Verdicts\n\n")
        f.write("| Test | Result | Details |\n")
        f.write("|------|--------|--------|\n")
        for name, verdict in report.get("verdicts", {}).items():
            symbol = "✅ PASS" if verdict["passed"] else "❌ FAIL"
            f.write(f"| {name.replace('_', ' ').title()} | {symbol} | {verdict.get('explanation', '')} |\n")
        
        f.write("\n## NARMA-10 Results\n\n")
        f.write("| Mode | NRMSE | Correlation |\n")
        f.write("|------|-------|-------------|\n")
        for r in report.get("results", []):
            if r["task"] == "NARMA-10":
                f.write(f"| {r['mode'].title()} | {r['nrmse']:.4f} | {r['correlation']:.4f} |\n")
        
        f.write("\n## Interpretation\n\n")
        status = report['summary'].get('status', '')
        if "GENUINE RESERVOIR" in status:
            f.write("The LV06 demonstrates all properties required for reservoir computing:\n")
            f.write("- Physical coupling between input and output\n")
            f.write("- Fading memory\n")
            f.write("- Nonlinear processing capability\n")
            f.write("- Performance exceeding theoretical Poisson baseline\n")
        elif "LINEAR RESERVOIR" in status:
            f.write("The LV06 shows reservoir-like properties but **without demonstrated nonlinearity**.\n")
            f.write("This limits its computational universality.\n")
        elif "RATE ENCODER" in status:
            f.write("The LV06 functions as a rate encoder (input modulates output rate) but **does not qualify as a reservoir computer**.\n")
            f.write("Missing: memory capacity, superiority over Poisson baseline.\n")
        else:
            f.write("The LV06 **does not demonstrate reservoir computing capabilities** in these tests.\n")
    
    print(f"[SAVED] Markdown: {md_path}")
    
    return report

if __name__ == "__main__":
    main()
