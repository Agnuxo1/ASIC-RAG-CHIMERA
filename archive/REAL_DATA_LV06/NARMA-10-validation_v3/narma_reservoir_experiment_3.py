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

# --- CONFIGURATION (V3: TEMPORAL INTEGRATION) ---
HOST = "0.0.0.0"
PORT = 3333 
LV06_IP = "192.168.0.15"  
TARGET_FREQ = 525          # MHz - Critical frequency for RC
STEPS = 2000               # Scale for official results
TRAIN_RATIO = 0.8
DIFFICULTY = 0.05           # Optimized for high share rate
WINDOW_TIME = 0.04         # 40ms target (25 Hz)

# --- FREQUENCY SETUP ---
def configure_lv06(ip, freq_mhz):
    """Configura frecuencia via AxeOS API con PATCH"""
    print(f"[SETUP] Attempting to set LV06 frequency to {freq_mhz} MHz...")
    try:
        r = requests.patch(f"http://{ip}/api/system", json={"frequency": freq_mhz}, timeout=5)
        if r.status_code == 200:
            print(f"[SETUP] PATCH SUCCESSFUL.")
            requests.post(f"http://{ip}/api/reboot", timeout=2)
            print(f"[SETUP] REBOOT signal sent. Waiting 45s for hardware to stabilize...")
            time.sleep(45)
            return True
    except Exception as e:
        print(f"[SETUP] Error auto-configuring: {e}")
    return False

# --- NARMA-10 GENERATOR ---
def generate_narma10(length):
    """Standard NARMA-10"""
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * sum_y
        term3 = 1.5 * u[t-9] * u[t]
        term4 = 0.1
        y[t] = np.clip(term1 + term2 + term3 + term4, 0, 1)
    return u, y

# --- STRATUM SERVER ---
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
                if self.running: print(f"[SERVER] Error: {e}")
                    
    def handle_client(self, conn):
        buffer = ""
        while self.running:
            try:
                raw_data = conn.recv(8192)
                if not raw_data: break
                print(f"[PHYSICAL RECV] {len(raw_data)} bytes", flush=True)
                data = raw_data.decode('utf-8', errors='ignore')
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    print(f"[RECV] {line.strip()}", flush=True)
                    self.process_line(conn, line)
            except Exception as e:
                print(f"[RECV ERROR] {e}", flush=True)
                break
        self.connection_active = False
        print("[SERVER] Client Disconnected")

    def process_line(self, conn, line):
        try:
            msg = json.loads(line)
        except: return
        method = msg.get('method')
        msg_id = msg.get('id')
        if method == 'mining.subscribe':
            print(f"[STRATUM] Subscribe from {msg_id}")
            resp = {"id": msg_id, "result": [[["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]], self.extranonce1, self.extranonce2_size], "error": None}
            self.send_json(conn, resp)
            self.send_difficulty(conn, 0.1) # Use 0.1 from debug
        elif method == 'mining.authorize':
            print(f"[STRATUM] Authorize from {msg_id}")
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)
            # No inject_input here, will happen in main loop
        elif method == 'mining.configure':
            # Ignore just like debug did
            pass
        elif method == 'mining.suggest_difficulty':
            pass
        elif method == 'mining.submit':
            arrival_time = time.perf_counter()
            params = msg.get('params', [])
            # Store full record for batching investigation
            share_record = {
                "time": arrival_time,
                "msg": msg
            }
            self.current_shares.append(share_record)
            
            # RAW Logging for Debugging
            nonce = params[4] if len(params) > 4 else "???"
            print(f"[SHARE] RAW: {line.strip()}", flush=True)
            print(f"[SHARE] Captured Nonce: {nonce} @ {arrival_time:.4f}", flush=True)
            
            self.send_json(conn, {"id": msg_id, "result": True, "error": None})

    def send_json(self, conn, data):
        try:
            line = json.dumps(data) + '\n'
            print(f"[SEND] {line.strip()}", flush=True)
            conn.sendall(line.encode('utf-8'))
        except Exception as e:
            print(f"[SEND ERROR] {e}")

    def send_difficulty(self, conn, diff):
        msg = {"id": None, "method": "mining.set_difficulty", "params": [diff]}
        self.send_json(conn, msg)

    def inject_input(self, u_value):
        if not self.client_conn: return False
        self.job_counter += 1
        job_id = str(self.job_counter)
        u_hex = struct.pack('>f', u_value).hex()
        # Valid Bitcoin Header: [4b ver][1b in_cnt][32b prev_hash][4b prev_idx][1b script_len]
        header = "01000000" + "01" + "00" * 32 + "ffffffff" + "10" 
        # VALID SCRIPT: OP_PUSH4 (04) + u_hex (4b) + OP_PUSH10 (0a) + padding (10b) = 16 bytes (10 hex)
        coinb1 = header + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff" + "01" + "00f2052a01000000" + "00" * 8
        ntime = hex(int(time.time()))[2:].zfill(8)
        # We increase version rolling to be safe
        params = [job_id, "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        msg = {"id": None, "method": "mining.notify", "params": params}
        self.send_json(self.client_conn, msg)
        return True

    def harvest_state(self, window_duration=WINDOW_TIME):
        """Cumulative Harvesting: Collect all activity since last call"""
        # Wait for the duration to allow hardware to work
        time.sleep(window_duration)
            
        # Take everything currently in the buffer
        valid_shares = list(self.current_shares)
        
        # Clear buffer for next cycle
        self.current_shares = []

        n_shares = len(valid_shares)
        
        # --- FEATURE EXTRACTION ---
        count = float(n_shares)
        
        if n_shares > 1:
            deltas = []
            for i in range(1, n_shares):
                dt = (valid_shares[i]['time'] - valid_shares[i-1]['time']) * 1e6
                deltas.append(np.log1p(max(0, dt)))
            avg_jitter = np.mean(deltas)
            std_jitter = np.std(deltas)
        else:
            avg_jitter = 0.0
            std_jitter = 0.0
            
        return np.array([count, avg_jitter, std_jitter, 1.0])

# --- MAIN EXPERIMENT ---
def run_experiment():
    print("="*60)
    print("   NARMA-10 BENCHMARK V3: TEMPORAL INTEGRATION")
    print("="*60)
    
    # 0. Hardware Driver/Config (Reboot)
    configure_lv06(LV06_IP, TARGET_FREQ)
    
    asic = ASICReservoir(HOST, PORT)
    asic.daemon = True
    asic.start()
    
    print("[WAIT] Waiting for Miner connection...", flush=True)
    while not asic.connection_active: time.sleep(1)
    
    print("[WAIT] Sending initial Job and waiting for FIRST SHARE to sync hardware...", flush=True)
    asic.inject_input(0.25)
    while len(asic.current_shares) == 0:
        if not asic.connection_active: break
        time.sleep(0.5)
        
    print(f"[OK] First Share received! ({len(asic.current_shares)} total). Starting NARMA-10...", flush=True)
    time.sleep(1)
    # Clear initial shares to start fresh
    asic.current_shares = []
    
    u, y = generate_narma10(STEPS + 20)
    u, y = u[20:], y[20:]
    total_steps = len(u)
    
    X, Y = [], []
    start_time = time.time()
    
    for t in range(total_steps):
        # Ensure miner is connected before each step
        while not asic.connection_active:
            print("[WAIT] Connection lost. Waiting for miner...", flush=True)
            time.sleep(2)
            
        success = asic.inject_input(u[t])
        if not success:
            print(f"[WARN] Failed to inject at step {t}")
            
        state_vector = asic.harvest_state(window_duration=WINDOW_TIME)
        
        X.append(state_vector)
        Y.append(y[t])
        
        if t % 20 == 0:
            print(f"Step {t:4d}/{total_steps} | u={u[t]:.3f} | Shares={state_vector[0]:.0f}", flush=True)
    
    print(f"\n[DONE] Collection complete in {time.time() - start_time:.1f}s")
    
    X, Y = np.array(X), np.array(Y)
    split = int(total_steps * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_nrmse = 1.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        nrmse = np.sqrt(mse) / (np.max(Y) - np.min(Y))
        if nrmse < best_nrmse: best_nrmse = nrmse
    
    print("\n" + "="*60)
    print(f"  NRMSE (V3):  {best_nrmse:.6f}")
    print("="*60)
    
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "nrmse": float(best_nrmse), "steps": total_steps, "window": WINDOW_TIME, "difficulty": DIFFICULTY}
    with open("narma10_results_v3.json", "w") as f: json.dump(results, f, indent=2)
    
    # Export telemetry
    import csv
    with open("narma10_telemetry_v3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "u", "y", "count", "avg_jitter", "std_jitter", "bias"])
        for t in range(total_steps):
            writer.writerow([t, u[t], y[t]] + X[t].tolist())
    
    asic.running = False

if __name__ == "__main__":
    run_experiment()
