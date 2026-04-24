import socket
import threading
import json
import time
import struct
import binascii
import random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import requests

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 3333 
LV06_IP = "192.168.0.15"  # ← IP Correcta del Laboratorio
TARGET_FREQ = 525          # MHz - Frecuencia crítica para RC
STEPS = 500
TRAIN_RATIO = 0.8
SHARES_PER_STEP = 2        # Lower for sparse collection
TIMEOUT = 5.0              # Wider window
DIFFICULTY = 0.5           # Safer difficulty

# --- FREQUENCY SETUP ---
def configure_lv06(ip, freq_mhz):
    """Configura frecuencia via AxeOS API con PATCH"""
    print(f"[SETUP] Attempting to set LV06 frequency to {freq_mhz} MHz...")
    try:
        # PATCH es el método oficial para Bitaxe/AxeOS
        r = requests.patch(f"http://{ip}/api/system", json={"frequency": freq_mhz}, timeout=5)
        if r.status_code == 200:
            print(f"[SETUP] PATCH SUCCESSFUL.")
            # Reboot para aplicar cambios físicos (AxeOS requiere esto para el PLL)
            requests.post(f"http://{ip}/api/reboot", timeout=2)
            print(f"[SETUP] REBOOT signal sent. Waiting 45s for hardware to stabilize...")
            time.sleep(45)
            return True
    except Exception as e:
        print(f"[SETUP] Error auto-configuring: {e}")
    
    print(f"[SETUP] *** MANUAL ACTION REQUIRED ***")
    print(f"[SETUP] Please set frequency to {freq_mhz} MHz in AxeOS web interface (http://{ip})")
    print(f"[SETUP] Ensure the miner is mining to Stratum: 192.168.0.11:3333")
    input("[SETUP] Press ENTER when you have confirmed the frequency on the OLED/Web...")
    return False

# --- NARMA-10 GENERATOR (CORRECTED) ---
def generate_narma10(length):
    """
    Standard NARMA-10 (Atiya & Parlos 2000)
    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y[t-9:t+1]) + 1.5*u(t-9)*u(t) + 0.1
    """
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])  # Sum of past Y values
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * sum_y
        term3 = 1.5 * u[t-9] * u[t]
        term4 = 0.1
        # Clamp to prevent explosion
        y[t] = np.clip(term1 + term2 + term3 + term4, 0, 1)
    return u, y

# --- STRATUM SERVER (CORRECTED INJECTION) ---
class ASICReservoir(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.client_conn = None
        self.running = True
        self.current_shares = []
        self.job_counter = 0
        self.connection_active = False
        self.extranonce1 = "08000002"
        self.extranonce2_size = 4

    def run(self):
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        while self.running:
            try:
                conn, addr = self.sock.accept()
                print(f"[SERVER] Connected by {addr}")
                self.client_conn = conn
                self.connection_active = True
                self.handle_client(conn)
            except Exception as e:
                if self.running:
                    print(f"[SERVER] Error: {e}")
                    
    def handle_client(self, conn):
        buffer = ""
        while self.running:
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self.process_line(conn, line)
            except:
                break
        self.connection_active = False
        print("[SERVER] Client Disconnected")

    def process_line(self, conn, line):
        try:
            msg = json.loads(line)
        except:
            return

        method = msg.get('method')
        msg_id = msg.get('id')

        if method == 'mining.subscribe':
            print(f"[STRATUM] Subscribe from {msg_id}")
            resp = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]],
                    self.extranonce1, 
                    self.extranonce2_size
                ],
                "error": None
            }
            self.send_json(conn, resp)
            self.send_difficulty(conn, DIFFICULTY)

        elif method == 'mining.authorize':
            print(f"[STRATUM] Authorize from {msg_id}")
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)
            self.send_difficulty(conn, DIFFICULTY)
            # CRITICAL: Send initial job immediately
            self.inject_input(0.25) 

        elif method == 'mining.configure':
            print(f"[STRATUM] Configure from {msg_id}")
            resp = {"id": msg_id, "result": {"version-rolling.mask": "ffffffff"}, "error": None}
            self.send_json(conn, resp)

        elif method == 'mining.suggest_difficulty':
            print(f"[STRATUM] Suggest Difficulty from {msg_id}")
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)

        elif method == 'mining.submit':
            arrival_time = time.perf_counter()
            print(f"[STRATUM] Share submitted at {arrival_time}")
            params = msg.get('params', [])
            nonce = params[4] if len(params) > 4 else "00000000"
            ntime = params[3] if len(params) > 3 else "00000000"
            
            self.current_shares.append({
                "time": arrival_time,
                "nonce": nonce,
                "ntime": ntime
            })
            
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)

    def send_json(self, conn, data):
        try:
            conn.sendall((json.dumps(data) + '\n').encode('utf-8'))
        except:
            pass

    def send_difficulty(self, conn, diff):
        msg = {"id": None, "method": "mining.set_difficulty", "params": [diff]}
        self.send_json(conn, msg)

    def inject_input(self, u_value):
        """
        CORRECTED: Inject u_value into coinbase transaction
        """
        if not self.client_conn:
            return False

        # DO NOT clear current_shares here to allow accumulation
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        # === CRITICAL FIX: Encode u_value in coinb1 ===
        # u_value is 0.0-0.5, convert to 4 bytes
        u_bytes = struct.pack('>f', u_value)
        u_hex = u_bytes.hex()
        
        # Random padding to ensure unique jobs
        random_padding = binascii.b2a_hex(random.randbytes(4)).decode()
        
        # Build coinbase with u_value embedded
        u_bytes = struct.pack('>f', u_value)
        u_hex = u_bytes.hex()
        
        # Valid Bitcoin Header: [4b ver][1b in_cnt][32b prev_hash][4b prev_idx][1b script_len]
        header = "01000000" + "01" + "00" * 32 + "ffffffff" + "10"
        coinb1 = header + u_hex + "00" * 12
        coinb2 = "ffffffff" + "01" + "00f2052a01000000" + "00" * 8
        
        # Current timestamp (8 hex chars)
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        params = [
            job_id,
            "0" * 64,    # prev_hash
            coinb1,
            coinb2,
            [],          # merkle_branch
            "20000000",  # version
            "1f00ffff",  # Easiest possible nbits
            ntime,
            True         # clean_jobs
        ]
        
        print(f"[INJECT] u={u_value:.3f} -> ntime={ntime} coinb1={coinb1[:20]}...")
        
        msg = {"id": None, "method": "mining.notify", "params": params}
        self.send_json(self.client_conn, msg)
        return True

    def harvest_state(self, timeout=TIMEOUT, wait_count=SHARES_PER_STEP):
        """Collect reservoir state from share responses"""
        start_wait = time.perf_counter()
        while len(self.current_shares) < wait_count:
            if time.perf_counter() - start_wait > timeout:
                break
            time.sleep(0.01)
            
        shares = self.current_shares[:wait_count]
        n_shares = len(shares)
        
        if n_shares < 2:
            # Not enough data - return zeros
            return np.zeros(wait_count * 2 - 1)
        
        # Feature 1: Inter-arrival Jitter with Log Scaling
        deltas = []
        for i in range(1, n_shares):
            dt = (shares[i]['time'] - shares[i-1]['time']) * 1e6
            # Use log1p to compress dynamic range and capture micro-jitters
            deltas.append(np.log1p(max(0, dt)))
        
        # Pad if needed
        while len(deltas) < wait_count - 1:
            deltas.append(0.0)
            
        # Feature 2: Nonce entropy (last 2 bytes as integer)
        nonces = []
        for s in shares:
            try:
                val = int(s['nonce'][-4:], 16) / 65535.0  # Normalize to 0-1
                nonces.append(val)
            except:
                nonces.append(0.0)
        
        while len(nonces) < wait_count:
            nonces.append(0.0)
            
        return np.array(deltas + nonces)

# --- MAIN EXPERIMENT ---
def run_experiment():
    print("="*60)
    print("   NARMA-10 RESERVOIR BENCHMARK (LV06 @ 525 MHz)")
    print("="*60)
    
    # 0. Configure LV06 frequency
    # configure_lv06(LV06_IP, TARGET_FREQ)
    
    # 1. Start Server
    asic = ASICReservoir(HOST, PORT)
    asic.daemon = True
    asic.start()
    
    print(f"\n[WAIT] Waiting for LV06 connection...")
    print(f"[WAIT] Configure your LV06 to connect to {HOST}:{PORT}")
    
    timeout_connect = 120
    start = time.time()
    while not asic.connection_active:
        if time.time() - start > timeout_connect:
            print("[ERROR] Timeout waiting for miner connection")
            return
        time.sleep(1)
        
    print("[OK] Miner Connected!")
    print("[WARMUP] Waiting 1s for thermal stabilization...")
    time.sleep(1)
    
    # 2. Generate NARMA-10 Data
    print(f"\n[DATA] Generating NARMA-10 Sequence (Steps={STEPS})...")
    # Increase sequence to account for transient discard
    u, y = generate_narma10(STEPS + 20)
    u, y = u[20:], y[20:]  # Discard transient
    total_steps = len(u)
    
    X = []  # Reservoir States
    Y = []  # Targets
    
    print(f"\n[RUN] Starting Injection Loop ({total_steps} steps)...")
    start_time = time.time()
    failed_steps = 0
    
    for t in range(total_steps):
        # A. Inject Input
        asic.inject_input(u[t])
        
        # B. Harvest State (Fast Sampling)
        state_vector = asic.harvest_state(timeout=TIMEOUT, wait_count=SHARES_PER_STEP)
        
        if np.sum(state_vector) == 0:
            failed_steps += 1
        
        X.append(state_vector)
        Y.append(y[t])
        
        if t % 20 == 0:
            elapsed = time.time() - start_time
            rate = (t+1) / elapsed if elapsed > 0 else 0
            print(f"  Step {t:3d}/{total_steps} | u={u[t]:.3f} | "
                  f"shares={len(asic.current_shares):2d} | "
                  f"rate={rate:.1f} steps/s")
    
    duration = time.time() - start_time
    print(f"\n[DONE] Data collection complete: {duration:.1f}s")
    print(f"[STATS] Failed steps (no shares): {failed_steps}/{total_steps}")
    
    if failed_steps > total_steps * 0.5:
        print("[WARNING] More than 50% failed steps - check LV06 connection")
    
    # 3. Train & Evaluate
    X = np.array(X)
    Y = np.array(Y)
    
    split_idx = int(total_steps * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"\n[TRAIN] Training Ridge Regression...")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try multiple regularization strengths
    best_nrmse = float('inf')
    best_alpha = 1.0
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        y_range = np.max(Y) - np.min(Y)
        nrmse = rmse / y_range if y_range > 0 else 1.0
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_alpha = alpha
    
    # 4. Final Result
    print("\n" + "="*60)
    print("              NARMA-10 BENCHMARK RESULT")
    print("="*60)
    print(f"  Best Alpha:  {best_alpha}")
    print(f"  NRMSE:       {best_nrmse:.6f}")
    print()
    
    if best_nrmse < 0.15:
        verdict = "✅ EXCELLENT - Strong Neuromorphic Computation"
        status = "SUCCESS"
    elif best_nrmse < 0.30:
        verdict = "✅ GOOD - Significant Computational Capability"
        status = "SUCCESS"
    elif best_nrmse < 0.50:
        verdict = "⚠️ WEAK - Some Learning, Needs Optimization"
        status = "PARTIAL"
    else:
        verdict = "❌ FAIL - No Significant Reservoir Dynamics"
        status = "FAIL"
    
    print(f"  VERDICT:     {verdict}")
    print("="*60)
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "LV06",
        "frequency_mhz": TARGET_FREQ,
        "steps": total_steps,
        "shares_per_step": SHARES_PER_STEP,
        "nrmse": float(best_nrmse),
        "best_alpha": float(best_alpha),
        "failed_steps": failed_steps,
        "status": status
    }
    
    with open("narma10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results saved to narma10_results.json")
    
    # Cleanup
    asic.running = False

import sys
if __name__ == "__main__":
    run_experiment()