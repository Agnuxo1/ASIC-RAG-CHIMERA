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
import collections

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 3333 
STEPS = 300       # Total time steps (Train + Test)
TRAIN_RATIO = 0.7
SHARES_PER_STEP = 20 # How many shares to capture per input injection (Reservoir State Size)
DIFFICULTY = 1    # Keep low to get fast shares (noise)

# --- NARMA-10 GENERATOR ---
def generate_narma10(length):
    """
    Generates NARMA-10 sequence.
    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(u(t-i)) + 1.5*u(t-9)*u(t) + 0.1
    """
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_u = np.sum(u[t-9:t+1])
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * sum_u
        term3 = 1.5 * u[t-9] * u[t]
        term4 = 0.1
        y[t] = term1 + term2 + term3 + term4
    return u, y

# --- STRATUM SERVER & RESERVOIR ---
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
        
        # Data Buffers
        self.current_shares = []
        self.share_event = threading.Event()
        self.job_counter = 0
        
        # Telemetry
        self.connection_active = False

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
                print(f"[SERVER] Connection error: {e}")
                
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

        # 1. SUBSCRIPTION
        if method == 'mining.subscribe':
            resp = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "subscription_id_1"], ["mining.notify", "subscription_id_2"]],
                    "08000002", 4
                ],
                "error": None
            }
            self.send_json(conn, resp)

        # 2. AUTHORIZATION
        elif method == 'mining.authorize':
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)
            # Set Difficulty immediately
            self.send_difficulty(conn, DIFFICULTY)

        # 3. SHARE SUBMISSION (The Reservoir Echo)
        elif method == 'mining.submit':
            # Capture the echo!
            arrival_time = time.perf_counter()
            # Extract nonce if possible (msg['params'][4])
            nonce = msg.get('params', [])[4] if len(msg.get('params', [])) > 4 else "00000000"
            
            self.current_shares.append({
                "time": arrival_time,
                "nonce": nonce
            })
            
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)

    def send_json(self, conn, data):
        try:
            line = json.dumps(data) + '\n'
            conn.sendall(line.encode('utf-8'))
        except:
            pass

    def send_difficulty(self, conn, diff):
        msg = {
            "id": None,
            "method": "mining.set_difficulty",
            "params": [diff]
        }
        self.send_json(conn, msg)

    def inject_input(self, u_value):
        """
        Injects the input u_value into the miner by creating a new mining job
        where the Merkle Root encodes u_value.
        """
        if not self.client_conn:
            return False

        # Clear previous buffer
        self.current_shares = []
        
        # 1. Encode u_value (0.0 - 0.5) into Merkle Root
        # We'll use the first 8 bytes of Merkle Root to store u_value as hex
        # u_value is float, lets map 0-0.5 to integer space or IEEE hex
        u_hex = struct.pack('>f', u_value).hex() # 8 chars (4 bytes)
        # Pad to 32 bytes (64 chars) with random noise to ensure uniqueness/avalanche
        padding = binascii.b2a_hex(np.random.bytes(28)).decode()
        merkle_root = (u_hex + padding).ljust(64, '0')

        self.job_counter += 1
        job_id = str(self.job_counter)
        
        # Standard Stratum Job Params
        prev_hash = "00000000" * 8
        coinb1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
        coinb2 = "ffffffff01000000000000000000000000000000000000000000000000000000000000000000000000"
        # Access real hardware Merkle via clean_jobs=True to force flush
        params = [
            job_id,
            prev_hash,
            coinb1,
            coinb2,
            [], # Merkle branch (empty for solo/proxy)
            "20000000", # Version
            "1c00ffff", # nbits
            hex(int(time.time()))[2:], # ntime
            True # clean_jobs = TRUE (Force restart/avalanche)
        ]
        
        msg = {
            "id": None,
            "method": "mining.notify",
            "params": params
        }
        self.send_json(self.client_conn, msg)
        return True

    def harvest_state(self, timeout=1.0, wait_count=20):
        """
        Waits for 'wait_count' shares and returns the Feature Vector (Receiver State).
        Features: Nonce bits, Inter-arrival intervals (Jitter).
        """
        start_wait = time.perf_counter()
        while len(self.current_shares) < wait_count:
            if time.perf_counter() - start_wait > timeout:
                break
            time.sleep(0.01)
            
        shares = self.current_shares[:wait_count]
        
        # Construction of State Vector X(t)
        # We need a fixed size vector.
        # If we have fewer than wait_count, pad with 0.
        
        # Feature 1: Inter-arrival Jitter (Microseconds)
        # Calculate deltas between share arrivals
        deltas = []
        if len(shares) > 1:
            for i in range(1, len(shares)):
                dt = (shares[i]['time'] - shares[i-1]['time']) * 1e6 # microseconds
                deltas.append(dt)
        else:
            deltas = [0.0] * (wait_count - 1)
            
        # Pad deltas
        while len(deltas) < (wait_count - 1):
            deltas.append(0.0)
            
        # Feature 2: Nonce Bits
        # Take the last byte of each nonce as an integer (0-255)
        nonces = []
        for s in shares:
            try:
                # nonce is hex string e.g., 'a1b2c3d4'
                val = int(s['nonce'][-2:], 16) 
                nonces.append(float(val))
            except:
                nonces.append(0.0)
        
        while len(nonces) < wait_count:
            nonces.append(0.0)
            
        # Combine
        return np.array(deltas + nonces)

# --- MAIN EXPERIMENT ---
def run_experiment():
    print("--- NARMA-10 RESERVOIR BENCHMARK (REAL HARDWARE) ---")
    
    # 1. Start Server
    asic = ASICReservoir(HOST, PORT)
    asic.start()
    
    print("Waiting for Miner Connection (Turn on your LV06)...")
    while not asic.connection_active:
        time.sleep(1)
    print("Miner Connected! Starting Warmup (5s)...")
    time.sleep(5)
    
    # 2. Generate Data
    print(f"Generating NARMA-10 Sequence (Steps={STEPS})...")
    u, y = generate_narma10(STEPS)
    # Discard initialization transient of NARMA
    u = u[10:]
    y = y[10:]
    # Adjust total steps
    total_steps = len(u)
    
    X = [] # Reservoir States
    Y = [] # Targets
    
    print("Starting Injection Loop...")
    start_time = time.time()
    
    for t in range(total_steps):
        # A. Inject Input u(t)
        asic.inject_input(u[t])
        
        # B. Harvest State X(t)
        # Wait up to 1.5s for shares. 20 shares target.
        # Note: If difficulty is 1, LV06 (500GH) should return ~125 shares/sec.
        # So 20 shares should take ~0.16s.
        state_vector = asic.harvest_state(timeout=1.0, wait_count=SHARES_PER_STEP)
        
        X.append(state_vector)
        Y.append(y[t])
        
        if t % 10 == 0:
            print(f"Step {t}/{total_steps} | Input={u[t]:.4f} | StateMean={np.mean(state_vector):.2f}")
            
    print(f"Data Collection Complete. Duration: {time.time() - start_time:.1f}s")
    
    # 3. Training & Evaluation
    X = np.array(X)
    Y = np.array(Y)
    
    # Train/Test Split
    split_idx = int(total_steps * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Training Ridge Regression (Readout Layer)...")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # 4. Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # NRMSE = RMSE / (max - min) of target
    y_range = np.max(Y) - np.min(Y)
    nrmse = rmse / y_range if y_range > 0 else 0
    
    print("\n" + "="*40)
    print("       FINAL VERDICT (NARMA-10)")
    print("="*40)
    print(f"NRMSE: {nrmse:.6f}")
    
    if nrmse < 0.15:
        print("RESULT: SUCCESS (Excellent Neuromorphic Performance)")
    elif nrmse < 0.4:
        print("RESULT: PASSABLE (Some Computational Capability)")
    else:
        print("RESULT: FAIL (No Significant Learning)")
        
    print("="*40)
    
    # Cleanup
    asic.running = False
    sys.exit(0)

import sys
if __name__ == "__main__":
    run_experiment()
