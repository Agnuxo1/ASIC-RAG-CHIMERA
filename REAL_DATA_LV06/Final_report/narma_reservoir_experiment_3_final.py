import socket
import json
import time
import struct
import binascii
import numpy as np
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 3333
LV06_IP = "192.168.0.15"
STEPS = 2000
STEPS = 2000
WINDOW_TIME = 0.04 # 25 Hz Target
DIFFICULTY = 512   # Stable hardware limit for LV06
TARGET_FREQ = 525

def generate_narma10(length):
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])
        y[t] = np.clip(0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1, 0, 1)
    return u, y

def configure_lv06(ip, freq):
    print(f"[HW] Setting frequency to {freq}MHz...")
    try:
        requests.patch(f"http://{ip}/api/system", json={"frequency": freq}, timeout=5)
        requests.post(f"http://{ip}/api/reboot", timeout=2)
        print("[HW] Rebooting... waiting 45s")
        time.sleep(45)
    except: print("[HW] Config failed, continuing...")

def run_benchmark():
    print("="*60)
    print("   NARMA-10 V3: ULTRA-FIDELITY SINGLE-THREADED")
    print("="*60)
    
    # configure_lv06(LV06_IP, TARGET_FREQ)
    
    u, y = generate_narma10(STEPS + 20)
    u, y = u[20:], y[20:]
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(120)
    
    print(f"[SRV] Listening on {PORT}...")
    conn, addr = server.accept()
    print(f"[SRV] Connected by {addr}")
    # Stay BLOCKING for handshake
    conn.setblocking(True) 
    
    buffer = ""
    current_shares = []
    
    def send_msg(msg):
        line = json.dumps(msg) + "\n"
        conn.sendall(line.encode())
        
    # --- HANDSHAKE ---
    start_handshake = time.time()
    authorized = False
    subscribed = False
    while not authorized and (time.time() - start_handshake < 30):
        try:
            data = conn.recv(4096).decode('utf-8', errors='ignore')
            if not data: break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                msg = json.loads(line)
                mid = msg.get('id')
                method = msg.get('method')
                print(f"[RECV] {method}")
                if method == 'mining.subscribe':
                    send_msg({"id": mid, "result": [[["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]], "08000002", 4], "error": None})
                    send_msg({"id": None, "method": "mining.set_difficulty", "params": [DIFFICULTY]})
                    subscribed = True
                elif method == 'mining.authorize':
                    send_msg({"id": mid, "result": True, "error": None})
                    authorized = True
                elif method == 'mining.configure':
                    send_msg({"id": mid, "result": {"version-rolling.mask": "ffffffff"}, "error": None})
        except BlockingIOError: time.sleep(0.1)
    
    if not authorized:
        print("[ERR] Handshake failed")
        return

    print("[OK] Authorized! Waiting for SYNC SHARE...")
    
    # Wait for first share to ensure hashing
    def inject(val, jid):
        u_hex = struct.pack('>f', val).hex()
        # Clean script: OP_PUSH4 + value + OP_PUSH10 + zeros
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + u_hex + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        send_msg({"id": None, "method": "mining.notify", "params": [str(jid), "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]}) # TRUE to keep miner in sync

    inject(0.25, 0)
    sync_share = False
    while not sync_share:
        try:
            data = conn.recv(4096).decode('utf-8', errors='ignore')
            if "\n" in data: sync_share = True
        except BlockingIOError: time.sleep(0.1)
    
    print("[OK] Hashing active! Starting Loop...")
    # NOW switch to non-blocking for high speed
    conn.setblocking(False) 
    
    X, Y = [], []
    start_time = time.time()
    
    for t in range(len(u)):
        # 1. Inject
        inject(u[t], t+1)
        
        # 2. Collect for WINDOW_TIME
        step_shares = []
        win_start = time.perf_counter()
        while (time.perf_counter() - win_start) < WINDOW_TIME:
            try:
                data = conn.recv(8192).decode('utf-8', errors='ignore')
                if data:
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            msg = json.loads(line)
                            if msg.get('method') == "mining.submit":
                                step_shares.append(time.perf_counter())
                                # Stratum ACK is MANDATORY
                                mid = msg.get('id')
                                send_msg({"id": mid, "result": True, "error": None})
                                print(".", end="", flush=True)
                        except: pass
            except BlockingIOError: pass
            time.sleep(0.001)
            
        # 3. Features
        n = len(step_shares)
        if n > 1:
            deltas = [np.log1p((step_shares[i] - step_shares[i-1])*1e6) for i in range(1, n)]
            state = [float(n), np.mean(deltas), np.std(deltas), 1.0]
        else:
            state = [float(n), 0.0, 0.0, 1.0]
            
        X.append(state)
        Y.append(y[t])
        
        if t % 50 == 0:
            print(f"\nStep {t}/{STEPS} | Shares={n} | u={u[t]:.3f}")

    print(f"\n[DONE] Time: {time.time()-start_time:.1f}s")
    
    # --- EVAL ---
    X, Y = np.array(X), np.array(Y)
    split = int(len(u)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_train)
    y_pred = model.predict(X_te_s)
    nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (np.max(Y) - np.min(Y))
    
    print("="*60)
    print(f" FINAL NRMSE: {nrmse:.6f}")
    print("="*60)
    
    with open("narma10_results_v3_final.json", "w") as f:
        json.dump({"nrmse": float(nrmse), "steps": len(u), "window": WINDOW_TIME, "diff": DIFFICULTY}, f)

if __name__ == "__main__":
    run_benchmark()
