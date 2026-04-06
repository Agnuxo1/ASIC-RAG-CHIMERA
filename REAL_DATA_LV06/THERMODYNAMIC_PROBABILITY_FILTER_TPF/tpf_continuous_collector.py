#!/usr/bin/env python3
"""
================================================================================
TPF CONTINUOUS TELEMETRY COLLECTOR
================================================================================
Scientific Objective:
Collect a high-volume dataset of real silicon signatures by running the miner
at very low difficulty (Constant 0.001). This ensures we get hundreds of 
"Success" signals (shares) to learn the physical footprint of a winning state.

Labeling:
Window-based. Every 2-second segment is labeled:
 - 1: Success (at least 1 share arrived)
 - 0: Failure (no shares arrived)
================================================================================
"""

import socket
import threading
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    DATASET_FILE = "tpf_continuous_dataset.csv"
    DURATION_SEC = 300 # 5 minutes of high-speed collection
    WINDOW_SEC = 2.0
    MIN_DIFF = 0.0001 # Extremely low to force share flood

class TPFContinuousCollector(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(1)
        
        self.conn = None
        self.authorized = False
        self.running = True
        self.share_log = [] # (arrival_time, job_id)
        
    def run(self):
        print(f"[SERVER] Listening on {self.config.PORT}...")
        try:
            self.conn, addr = self.sock.accept()
            print(f"[SERVER] Miner connected: {addr}")
            self.conn.settimeout(0.5)
            buffer = ""
            while self.running:
                try:
                    data = self.conn.recv(4096).decode('utf-8', errors='ignore')
                    if not data: break
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self._handle_msg(json.loads(line))
                except socket.timeout: continue
                except: break
        except Exception as e:
            print(f"[ERR] Server: {e}")

    def _handle_msg(self, msg):
        method = msg.get('method')
        mid = msg.get('id')
        if method == 'mining.subscribe':
            res = [[["mining.notify","1"]], "00", 4]
            self._send({"id": mid, "result": res, "error": None})
        elif method == 'mining.authorize':
            self._send({"id": mid, "result": True, "error": None})
            self.authorized = True
        elif method == 'mining.submit':
            self.share_log.append((time.perf_counter(), msg['params'][1]))
            # safe_print would be better but keeping it simple
            self._send({"id": mid, "result": True, "error": None})

    def _send(self, data):
        if self.conn:
            self.conn.sendall((json.dumps(data) + '\n').encode())

    def set_difficulty(self, diff):
        self._send({"id": None, "method": "mining.set_difficulty", "params": [diff]})

    def notify(self, clean=True):
        job_id = f"job_{int(time.perf_counter()*1000)}"
        params = [job_id, "0"*64, "0"*70, "0"*20, [], "20000000", "1f00ffff", hex(int(time.time()))[2:], clean]
        self._send({"id": None, "method": "mining.notify", "params": params})
        return job_id

def main():
    config = Config()
    collector = TPFContinuousCollector(config)
    collector.start()
    
    while not collector.authorized:
        print("[WAIT] Connect Miner...")
        time.sleep(2)

    print("\n" + "="*60)
    print("   TPF CONTINUOUS DATA COLLECTION (High Efficiency)")
    print("="*60)
    
    collector.set_difficulty(config.MIN_DIFF)
    collector.notify()
    
    start_time = time.perf_counter()
    dataset = []
    
    # Feature extraction state
    prev_share_count = 0
    
    print(f"[START] Collecting for {config.DURATION_SEC}s...")
    
    while (time.perf_counter() - start_time) < config.DURATION_SEC:
        loop_start = time.perf_counter()
        time.sleep(config.WINDOW_SEC)
        
        # Capture current signature
        now = time.perf_counter()
        recent_shares = [s for s in collector.share_log if loop_start <= s[0] < now]
        count = len(recent_shares)
        
        # Label: 1 if ANY share arrived in this window
        label = 1 if count > 0 else 0
        
        # Jitter of shares *in this window*
        jitters = [abs(s[0] - loop_start) for s in recent_shares]
        
        features = {
            "timestamp": datetime.now().isoformat(),
            "window_shares": count,
            "jitter_mean": np.mean(jitters) if count > 0 else 0,
            "jitter_std": np.std(jitters) if count > 1 else 0,
            "label": label
        }
        dataset.append(features)
        
        elapsed = now - start_time
        print(f"   [{elapsed:4.1f}s] Window Shares: {count} | Total: {len(collector.share_log)} | Label: {label}")
        
        # Every 30s send a new job to prevent stale work
        if int(elapsed) % 30 == 0:
            collector.notify(clean=True)

    # Save
    df = pd.DataFrame(dataset)
    df.to_csv(config.DATASET_FILE, index=False)
    print(f"\n[DONE] Dataset saved: {config.DATASET_FILE} ({len(df)} windows, {len(df[df['label']==1])} successes)")
    
    collector.running = False
    print("Experiment Finished.")

if __name__ == "__main__":
    main()
