import sys
import os
import time
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add the SDK path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "LV06_Universal_Config_Guide")))
from universal_lv06_driver import LV06StratumServer

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 3333
STEPS = 500
WINDOW_TIME = 1.0   # 1Hz sampling
D_BASE = 25.0       # Calibrated for ~10 Hz share rate
EPSILON = 0.05

def generate_narma10(length):
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0,t-9):t+1])
        y[t] = np.clip(0.3*y[t-1] + 0.05*y[t-1]*sum_y + 1.5*u[t-9]*u[t] + 0.1, 0, 1)
    return u, y

def run_rerc_validation(mode="normal"):
    print(f"\n[RERC] Starting Validation Mode: {mode.upper()}")
    u, y = generate_narma10(STEPS + 20)
    u, y = u[20:], y[20:]
    
    server = LV06StratumServer(HOST, PORT)
    server.daemon = True
    server.start()
    
    print("[SRV] Waiting for Miner...")
    while not server.connection_active:
        time.sleep(1)
    
    print("[OK] Connected. Starting Experiment...")
    
    X, Y = [], []
    start_time = time.time()
    
    for t in range(len(u)):
        # INJECTION
        if mode == "constant":
            server.inject_rate(0.25, d_base=D_BASE, epsilon=EPSILON)
        else:
            server.inject_rate(u[t], d_base=D_BASE, epsilon=EPSILON)
            
        # HARVESTING
        time.sleep(WINDOW_TIME)
        shares = server.harvest_state()
        
        # FEATURES (Rate-Encoding focused)
        n = len(shares)
        if n > 0:
            avg_iat = WINDOW_TIME / n # Mean Inter-Arrival Time
            state = [float(n), avg_iat, 1.0]
        else:
            state = [0.0, WINDOW_TIME, 1.0]
            
        X.append(state)
        Y.append(y[t])
        
        if t % 50 == 0:
            print(f" Step {t}/{len(u)} | Input={u[t]:.3f} | Shares={n}")
            
    server.stop()
    X, Y = np.array(X), np.array(Y)
    
    if mode == "shuffle":
        print(" [X] Shuffling X-Y pairs...")
        idx = np.random.permutation(len(X))
        X = X[idx]
        
    # Ridge Evaluation
    split = int(len(u)*0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = Y[:split], Y[split:]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    nrmse = np.sqrt(mean_squared_error(y_te, y_pred)) / (np.max(Y) - np.min(Y))
    
    return nrmse

if __name__ == "__main__":
    print("="*60)
    print(" RATE-ENCODED RESERVOIR COMPUTING (RE-RC) BENCHMARK")
    print("="*60)
    
    results = {}
    # We do them sequentially to avoid port conflicts
    results["normal"] = run_rerc_validation(mode="normal")
    time.sleep(5)
    results["shuffle"] = run_rerc_validation(mode="shuffle")
    time.sleep(5)
    results["constant"] = run_rerc_validation(mode="constant")
    
    print("\n" + "="*40)
    print(" RE-RC RIGOR RESULTS")
    print("="*40)
    for k, v in results.items():
        print(f" {k.upper():10}: NRMSE = {v:.6f}")
    print("="*40)
    
    if results["normal"] < results["shuffle"] * 0.9 and results["normal"] < results["constant"] * 0.9:
        print("\n [SUCCESS] GENUINE PHYSICAL COUPLING VERIFIED")
    else:
        print("\n [FAILURE] PHYSICAL COUPLING INSUFFICIENT")
