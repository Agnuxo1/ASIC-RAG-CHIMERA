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
    WINDOW_TIME: float = 2.0        # <--- STABLE: 2.0s to prevent pipeline saturation
    HANDSHAKE_TIMEOUT: float = 600.0 # 10 minutes
    WARMUP_STEPS: int = 15          
    
    # Rate Encoding Parameters
    D_BASE: float = 1.0             # <--- SENSIBLE: 1.0 to capture dynamics in 2s windows
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
        """
        if not self.client_conn or not self.authorized:
            return False
            
        # Calculate rate-encoded difficulty
        D = self.config.D_BASE / (u_value + self.config.EPSILON)
        
        # Send new difficulty
        self._send_difficulty(self.client_conn, D)
        
        # Delay to ensure difficulty is processed
        time.sleep(0.02)
        
        # Send new job
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        u_hex = struct.pack('>f', u_value).hex()
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
        with self.buffer_lock:
            shares = list(self.share_buffer)
            if clear:
                self.share_buffer = []
        return shares
        
    def stop(self):
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
    n = len(shares)
    if n == 0: return [0.0, window_time * 1000, 0.0, 0.0, 0.0, 1.0]
    rate = n / window_time
    if n == 1: return [float(n), window_time * 1000, 0.0, rate, 0.0, 1.0]
    times = [s['time'] for s in shares]
    iats = [(times[i] - times[i-1]) * 1000 for i in range(1, n)]
    mean_iat, std_iat = mean(iats), std(iats)
    cv = std_iat / mean_iat if mean_iat > 0 else 0.0
    return [float(n), mean_iat, std_iat, rate, cv, 1.0]

# =============================================================================
# SIMULATED POISSON BASELINE
# =============================================================================
def simulate_poisson_shares(u_values: List[float], config: ExperimentConfig, seed: int = 42) -> List[List[float]]:
    random.seed(seed)
    k = 10.0 * config.D_BASE
    features = []
    for u in u_values:
        D = config.D_BASE / (u + config.EPSILON)
        lambda_u = k / D
        arrival_times = []
        t = 0.0
        while t < config.WINDOW_TIME:
            iat = random.expovariate(lambda_u) if lambda_u > 0 else config.WINDOW_TIME
            t += iat
            if t < config.WINDOW_TIME: arrival_times.append(t)
        n = len(arrival_times)
        if n == 0: feat = [0.0, config.WINDOW_TIME * 1000, 0.0, 0.0, 0.0, 1.0]
        elif n == 1: feat = [float(n), config.WINDOW_TIME * 1000, 0.0, n / config.WINDOW_TIME, 0.0, 1.0]
        else:
            iats = [(arrival_times[i] - arrival_times[i-1]) * 1000 for i in range(1, n)]
            feat = [float(n), mean(iats), std(iats), n / config.WINDOW_TIME, std(iats)/mean(iats), 1.0]
        features.append(feat)
    return features

# (Continued Script content)
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
            
            print(f"\n  [{mode.upper()}] NRMSE = {nrmse_score:.4f} | r = {corr:.4f} | alpha = {best_alpha}")

    def _determine_status(self, verdicts: Dict) -> str:
        """Determine overall status based on verdicts"""
        passed = [k for k, v in verdicts.items() if v["passed"]]
        rc_requirements = ["rate_encoding_coupling", "beats_poisson_baseline", "memory_capacity"]
        rc_passed = all(k in passed for k in rc_requirements if k in verdicts)
        
        if rc_passed and "nonlinear_processing" in passed:
            return "GENUINE RESERVOIR COMPUTER - All criteria met"
        elif rc_passed:
            return "LINEAR RESERVOIR - RC criteria met but no nonlinearity demonstrated"
        elif "rate_encoding_coupling" in passed:
            return "RATE ENCODER - Physical coupling exists but not a full RC"
        else:
            return "NO RC CAPABILITY - Hardware does not function as reservoir"

def main():
    experiment = DefinitiveExperiment()
    report = experiment.run_all()
    print(f"\nStatus: {report['summary'].get('status', 'Unknown')}")
    print(f"Tests Passed: {report['summary'].get('passed', 0)}/{report['summary'].get('total', 0)}")

if __name__ == "__main__":
    main()
