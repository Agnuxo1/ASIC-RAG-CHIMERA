import socket
import json
import time
import struct
import binascii
import numpy as np
import requests
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 3333
LV06_IP = "192.168.0.15"
STEPS = 500  # Faster runs for validation
WINDOW_TIME = 0.08 
DIFFICULTY = 1.0 # Standard difficulty
TARGET_FREQ = 525

def generate_narma10(length):
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])
        y[t] = np.clip(0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1, 0, 1)
    return u, y

class RigorValidator:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.server.settimeout(60)
        self.conn = None
        self.buffer = ""
        self.job_counter = 0

    def send_msg(self, msg):
        line = json.dumps(msg) + "\n"
        self.conn.sendall(line.encode())

    def handshake(self):
        print("[SRV] Waiting for miner...")
        self.conn, addr = self.server.accept()
        print(f"[SRV] Connected by {addr}")
        self.conn.setblocking(True)
        
        start = time.time()
        authorized = False
        while not authorized and (time.time() - start < 30):
            data = self.conn.recv(4096).decode('utf-8', errors='ignore')
            self.buffer += data
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                msg = json.loads(line)
                method = msg.get('method')
                mid = msg.get('id')
                if method == 'mining.subscribe':
                    self.send_msg({"id": mid, "result": [[["mining.set_difficulty", "s1"], ["mining.notify", "s2"]], "08000002", 4], "error": None})
                    self.send_msg({"id": None, "method": "mining.set_difficulty", "params": [DIFFICULTY]})
                elif method == 'mining.authorize':
                    self.send_msg({"id": mid, "result": True, "error": None})
                    authorized = True
        return authorized

    def inject(self, val, jid, constant_mode=False):
        # IF constant_mode=True, we use 0.25 regardless of input
        actual_val = 0.25 if constant_mode else val
        u_hex = struct.pack('>f', actual_val).hex()
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        self.send_msg({"id": None, "method": "mining.notify", "params": [str(jid), "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]})

    def run_experiment(self, u, y, mode="normal"):
        print(f"\n>>> MODE: {mode.upper()}")
        self.conn.setblocking(False)
        X, Y = [], []
        
        for t in range(len(u)):
            self.inject(u[t], t+1, constant_mode=(mode=="constant"))
            
            step_shares = []
            win_start = time.perf_counter()
            while (time.perf_counter() - win_start) < WINDOW_TIME:
                try:
                    data = self.conn.recv(8192).decode('utf-8', errors='ignore')
                    if data:
                        self.buffer += data
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            if "mining.submit" in line:
                                step_shares.append(time.perf_counter())
                                # ACK
                                msg = json.loads(line)
                                self.send_msg({"id": msg.get('id'), "result": True, "error": None})
                except BlockingIOError: pass
                time.sleep(0.001)
            
            n = len(step_shares)
            if n > 1:
                deltas = [np.log1p((step_shares[i] - step_shares[i-1])*1e6) for i in range(1, n)]
                state = [float(n), np.mean(deltas), np.std(deltas), 1.0]
            else:
                state = [float(n), 0.0, 0.0, 1.0]
            
            X.append(state)
            Y.append(y[t])
            if t % 100 == 0: print(f" Step {t}/{len(u)}")

        X, Y = np.array(X), np.array(Y)
        
        if mode == "shuffle":
            print(" [X] Shuffling X-Y pairs to break causal link...")
            idx = np.random.permutation(len(X))
            X = X[idx]

        # Ridge Eval
        split = int(len(u)*0.8)
        model = Ridge(alpha=1.0)
        X_tr = X[:split]; X_te = X[split:]; y_tr = Y[:split]; y_te = Y[split:]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        nrmse = np.sqrt(mean_squared_error(y_te, y_pred)) / (np.max(Y) - np.min(Y))
        return nrmse

def main():
    u, y = generate_narma10(STEPS + 20)
    u, y = u[20:], y[20:]
    
    val = RigorValidator()
    if not val.handshake(): 
        print("Handshake failed")
        return

    results = {}
    results["normal"] = val.run_experiment(u, y, mode="normal")
    results["shuffle"] = val.run_experiment(u, y, mode="shuffle")
    results["constant"] = val.run_experiment(u, y, mode="constant")
    
    print("\n" + "="*40)
    print(" SCIENTIFIC RIGOR RESULTS")
    print("="*40)
    for k, v in results.items():
        print(f" {k.upper():10}: NRMSE = {v:.6f}")
    print("="*40)
    
    if results["normal"] < results["shuffle"] * 0.8:
        print("\n [CONCLUSION] REAL PHYSICAL COUPLING DETECTED")
    else:
        print("\n [CONCLUSION] SPURIOUS CORRELATION LIKELY (Coupling Failed)")

if __name__ == "__main__":
    main()
