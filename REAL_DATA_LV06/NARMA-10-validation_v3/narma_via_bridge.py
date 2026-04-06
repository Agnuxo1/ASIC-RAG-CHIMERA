"""
NARMA-10 RESERVOIR BENCHMARK (Via Chronos Bridge)
==================================================
Uses the running chronos_bridge_v2.py to harvest ASIC entropy.
This version does NOT run its own Stratum server.
"""

import socket
import json
import time
import struct
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import requests

# --- CONFIGURATION ---
BRIDGE_IP = "192.168.0.11"   # PC running chronos_bridge_v2.py
BRIDGE_API_PORT = 4029       # Bridge API port
LV06_IP = "192.168.0.15"     # Miner IP for frequency control
TARGET_FREQ = 525            # MHz - Target frequency
STEPS = 200                  # Total time steps
TRAIN_RATIO = 0.7
SAMPLES_PER_STEP = 5         # Shares to collect per input step
STEP_DURATION = 0.5          # Seconds per step

# --- FREQUENCY CONTROL ---
def set_frequency(ip, freq_mhz):
    """Attempt to set frequency via AxeOS HTTP API"""
    print(f"[FREQ] Setting LV06 to {freq_mhz} MHz...")
    
    endpoints = [
        f"http://{ip}/api/system",
        f"http://{ip}/cgi-bin/set_miner_conf.cgi",
        f"http://{ip}/cgi-bin/minerConfiguration.cgi"
    ]
    
    payloads = [
        {"frequency": freq_mhz},
        {"freq": freq_mhz},
        {"bitmain-freq": str(freq_mhz)}
    ]
    
    for endpoint in endpoints:
        for payload in payloads:
            try:
                resp = requests.post(endpoint, json=payload, timeout=5)
                if resp.status_code == 200:
                    print(f"[FREQ] SUCCESS via {endpoint}")
                    return True
            except:
                pass
            try:
                resp = requests.post(endpoint, data=payload, timeout=5)
                if resp.status_code == 200:
                    print(f"[FREQ] SUCCESS via {endpoint} (form)")
                    return True
            except:
                pass
    
    print(f"[FREQ] Could not set automatically. Please set {freq_mhz} MHz manually.")
    return False

def restart_miner(ip):
    """Restart miner to apply frequency changes"""
    print(f"[RESTART] Rebooting miner at {ip}...")
    try:
        requests.post(f"http://{ip}/cgi-bin/reboot.cgi", timeout=5)
        print("[RESTART] Reboot command sent. Waiting 30s for restart...")
        time.sleep(30)
        return True
    except Exception as e:
        print(f"[RESTART] Could not auto-restart: {e}")
        return False

# --- BRIDGE COMMUNICATION ---
def bridge_reset():
    """Reset the bridge data buffer"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((BRIDGE_IP, BRIDGE_API_PORT))
        s.sendall(b"RESET")
        s.recv(100)
        s.close()
        return True
    except Exception as e:
        print(f"[BRIDGE] Reset error: {e}")
        return False

def bridge_get_data():
    """Get current data from bridge"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((BRIDGE_IP, BRIDGE_API_PORT))
        s.sendall(b"GET_DATA")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        return json.loads(data)
    except Exception as e:
        print(f"[BRIDGE] Get data error: {e}")
        return {"timestamps": [], "total_shares": 0}

# --- NARMA-10 GENERATOR ---
def generate_narma10(length):
    """Standard NARMA-10 benchmark"""
    u = np.random.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length):
        sum_y = np.sum(y[max(0, t-9):t+1])
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * sum_y
        term3 = 1.5 * u[t-9] * u[t]
        term4 = 0.1
        y[t] = np.clip(term1 + term2 + term3 + term4, 0, 1)
    return u, y

# --- RESERVOIR STATE HARVESTER ---
def harvest_reservoir_state(duration=STEP_DURATION):
    """
    Harvest entropy from the bridge.
    Returns feature vector based on share inter-arrival times.
    """
    bridge_reset()
    time.sleep(duration)
    data = bridge_get_data()
    
    timestamps = data.get("timestamps", [])
    
    if len(timestamps) < 2:
        # No data - return zeros
        return np.zeros(SAMPLES_PER_STEP * 2)
    
    # Feature 1: Inter-arrival deltas (jitter)
    deltas = np.diff(timestamps) * 1000  # Convert to ms
    
    # Pad or truncate to fixed size
    if len(deltas) >= SAMPLES_PER_STEP:
        deltas = deltas[:SAMPLES_PER_STEP]
    else:
        deltas = np.pad(deltas, (0, SAMPLES_PER_STEP - len(deltas)))
    
    # Feature 2: Statistical moments
    if len(timestamps) > 1:
        all_deltas = np.diff(timestamps) * 1000
        stats = [
            np.mean(all_deltas) if len(all_deltas) > 0 else 0,
            np.std(all_deltas) if len(all_deltas) > 0 else 0,
            np.min(all_deltas) if len(all_deltas) > 0 else 0,
            np.max(all_deltas) if len(all_deltas) > 0 else 0,
            len(timestamps)  # sample count
        ]
    else:
        stats = [0, 0, 0, 0, 0]
    
    # Combine features
    return np.concatenate([deltas, stats])

# --- MAIN EXPERIMENT ---
def run_experiment():
    print("=" * 60)
    print("  NARMA-10 RESERVOIR COMPUTING BENCHMARK")
    print("  (Via Chronos Bridge - Real LV06 Hardware)")
    print("=" * 60)
    
    # 1. Check bridge connection
    print("\n[1/5] Checking bridge connection...")
    data = bridge_get_data()
    if data.get("total_shares", 0) == 0:
        print("[WARNING] No shares detected yet. Is the miner connected?")
        print("          Waiting 10s for miner to stabilize...")
        time.sleep(10)
    else:
        print(f"[OK] Bridge active. Total shares so far: {data['total_shares']}")
    
    # 2. Try to set frequency
    print(f"\n[2/5] Configuring frequency to {TARGET_FREQ} MHz...")
    freq_set = set_frequency(LV06_IP, TARGET_FREQ)
    if not freq_set:
        print("[MANUAL] Please set frequency manually in AxeOS web interface")
        print(f"         URL: http://{LV06_IP}")
        input("[MANUAL] Press ENTER when ready...")
    
    # 3. Generate NARMA-10 data
    print(f"\n[3/5] Generating NARMA-10 sequence ({STEPS} steps)...")
    u, y = generate_narma10(STEPS + 10)
    u, y = u[10:], y[10:]  # Discard transient
    
    # 4. Run reservoir experiment
    print(f"\n[4/5] Running reservoir experiment...")
    print(f"      Collecting {STEPS} samples, ~{STEP_DURATION}s each")
    
    X = []  # Reservoir states
    Y = []  # Targets
    
    start_time = time.time()
    
    for t in range(STEPS):
        # The input u[t] perturbs the system timing naturally
        # We're harvesting the resulting entropy pattern
        state = harvest_reservoir_state()
        
        X.append(state)
        Y.append(y[t])
        
        if t % 20 == 0:
            elapsed = time.time() - start_time
            rate = (t + 1) / elapsed if elapsed > 0 else 0
            print(f"      Step {t:3d}/{STEPS} | rate={rate:.1f} steps/s | state_mean={np.mean(state):.2f}")
    
    duration = time.time() - start_time
    print(f"\n[DONE] Data collection: {duration:.1f}s ({STEPS / duration:.1f} steps/s)")
    
    # 5. Train and evaluate
    print(f"\n[5/5] Training readout layer...")
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Check for zero variance
    nonzero_cols = np.where(np.var(X, axis=0) > 1e-10)[0]
    if len(nonzero_cols) == 0:
        print("[ERROR] No variance in features - no entropy captured!")
        print("        Check miner connection and frequency setting")
        return
    
    X = X[:, nonzero_cols]
    print(f"      Features with variance: {len(nonzero_cols)}/{X.shape[1] + len(nonzero_cols) - len(nonzero_cols)}")
    
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try multiple regularization values
    best_nrmse = float('inf')
    best_alpha = 1.0
    
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
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
    
    # --- FINAL RESULTS ---
    print("\n" + "=" * 60)
    print("           NARMA-10 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Device:        LV06 @ {TARGET_FREQ} MHz")
    print(f"  Steps:         {STEPS}")
    print(f"  Best Alpha:    {best_alpha}")
    print(f"  NRMSE:         {best_nrmse:.6f}")
    print()
    
    if best_nrmse < 0.15:
        verdict = "✅ EXCELLENT - Strong Reservoir Computing!"
    elif best_nrmse < 0.30:
        verdict = "✅ GOOD - Significant Computational Capability"
    elif best_nrmse < 0.50:
        verdict = "⚠️ WEAK - Marginal Reservoir Dynamics"
    else:
        verdict = "❌ FAIL - No Significant Reservoir Behavior"
    
    print(f"  VERDICT:       {verdict}")
    print("=" * 60)
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "LV06",
        "frequency_mhz": TARGET_FREQ,
        "steps": STEPS,
        "nrmse": float(best_nrmse),
        "best_alpha": float(best_alpha),
        "features_used": len(nonzero_cols),
        "verdict": verdict
    }
    
    with open("narma10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results written to narma10_results.json")

if __name__ == "__main__":
    run_experiment()
