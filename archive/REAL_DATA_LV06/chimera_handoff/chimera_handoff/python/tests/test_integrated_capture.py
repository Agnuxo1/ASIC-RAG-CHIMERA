#!/usr/bin/env python3
import sys
import os
import time
import threading
import json
import socket
from pathlib import Path

# Setup Path
BASE_DIR = Path("D:/ASIC_RAG/REAL_DATA_LV06/chimera_handoff")
SRC_DIR = BASE_DIR / "chimera_handoff" / "python" / "src"
sys.path.append(str(SRC_DIR))

from chimera_handoff.experiments.chimera_medical_handoff import Config, IntegratedHandoffBridge

def mock_client_thread():
    print(" [MOCK] Client thread started, waiting for bridge...")
    time.sleep(2) # Wait for bridge
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 3333))
        print(" [MOCK] Connected to bridge")
        s.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
        r1 = s.recv(1024).decode()
        print(f" [MOCK] Subscribe response: {r1}")
        s.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": ["u", "p"]}).encode() + b"\n")
        r2 = s.recv(1024).decode()
        print(f" [MOCK] Authorize response: {r2}")
        
        # Respond to one job
        print(" [MOCK] Waiting for job...")
        data = s.recv(4096).decode()
        print(f" [MOCK] Job received: {data[:100]}...")
        for line in data.split("\n"):
            if "mining.notify" in line:
                msg = json.loads(line)
                jid = msg["params"][0]
                print(f" [MOCK] Found notify for job {jid}, submitting share...")
                s.sendall(json.dumps({"id": 10, "method": "mining.submit", "params": ["u", jid, "0", "0", "n"]}).encode() + b"\n")
                r3 = s.recv(1024).decode()
                print(f" [MOCK] Submit response: {r3}")
                break
        time.sleep(2)
    except Exception as e:
        print(f" [MOCK] ERROR: {e}")
    finally:
        s.close()
        print(" [MOCK] Client closed")

def run_test():
    out_dir = Path("D:/ASIC_RAG/REAL_DATA_LV06/chimera_handoff/runs/test_capture")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        
    config = Config()
    config.DURATION_SEC = 10
    
    bridge = IntegratedHandoffBridge(config, out_dir)
    print(" [TEST] Starting bridge...")
    bridge.start()
    
    # Start mock client
    client = threading.Thread(target=mock_client_thread)
    client.start()
    
    # Run bridge loop for a few seconds
    print(" [TEST] Waiting for authorization...")
    start = time.perf_counter()
    timeout = 15
    while time.perf_counter() - start < timeout:
        if bridge.authorized:
            print(" [TEST] Authorized! Sealing batch...")
            if bridge.seal_batch_and_wait(config.D_BASE, timeout=5.0):
                print(" [TEST] share CAPTURED in bridge!")
                break
            else:
                print(" [TEST] seal_batch_and_wait TIMEOUT")
        else:
            time.sleep(0.5)
            
    bridge.save_artifacts()
    bridge.running = False
    print(f" [TEST] Artifacts saved, events count: {len(bridge.events)}")
    
    # VERIFICATION
    print("\n🧐 VERIFYING ARTIFACTS...")
    events_file = out_dir / "events.csv"
    protocol_file = out_dir / "protocol.json"
    metrics_file = out_dir / "application_metrics.json"
    
    if events_file.exists():
        print(f"✅ events.csv created ({events_file.stat().st_size} bytes)")
        with open(events_file, "r") as f:
            header = f.readline().strip()
            print(f"   Header: {header}")
    else:
        print("❌ events.csv MISSING")
        
    if protocol_file.exists():
        print("✅ protocol.json created")
    else:
        print("❌ protocol.json MISSING")
        
    if metrics_file.exists():
        print("✅ application_metrics.json created")
        with open(metrics_file, "r") as f:
            print(f"   Metrics: {f.read()}")
    else:
        print("❌ application_metrics.json MISSING")

if __name__ == "__main__":
    run_test()
