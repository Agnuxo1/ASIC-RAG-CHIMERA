#!/usr/bin/env python3
"""
================================================================================
TPF TELEMETRY COLLECTOR & BLOCKCHAIN REPLAY
================================================================================
Scientific Objective:
Collect a high-fidelity dataset of silicon signatures from the LV06. 
Includes "Golden Samples" by replaying historical winning blocks from the 
Bitcoin blockchain to observe how the ASIC's reservoir looks under success.

Modes:
1. REPLAY: Set the miner to process known winning block headers.
2. STREAM: Normal operation to capture high-entropy "loser" signatures.
================================================================================
"""

import socket
import threading
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    MINER_IP = "192.168.0.15"
    DATASET_FILE = "tpf_balanced_dataset.csv"
    REPLAY_BLOCKS = 20 # Increased for better sample size
    LOSER_STEPS = 100 # Increased for better baseline

class TPFCollector(threading.Thread):
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
        self.dataset = []
        self.share_log = [] # (arrival_time, job_id)
        
    def run(self):
        print(f"[SERVER] Listening for LV06 on {self.config.PORT}...")
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
            self._send({"id": mid, "result": True, "error": None})

    def _send(self, data):
        if self.conn:
            self.conn.sendall((json.dumps(data) + '\n').encode())

    def inject_job(self, diff, header_data=None):
        """Injects a mining job. If header_data is provided, it's a REPLAY."""
        self._send({"id": None, "method": "mining.set_difficulty", "params": [diff]})
        job_id = f"job_{int(time.perf_counter()*1000)}"
        
        # Stratum Notify: [job_id, prevhash, coinb1, coinb2, merkle, version, nbits, ntime, clean]
        if header_data:
            params = [job_id] + header_data
        else:
            # Random "Loser" Job
            params = [job_id, "0"*64, "0"*70, "0"*20, [], "20000000", "1f00ffff", hex(int(time.time()))[2:], True]
        
        self._send({"id": None, "method": "mining.notify", "params": params})
        return job_id, time.perf_counter()

    def capture_signature(self, t_sent, label):
        """Extracts jitter and entropy features from the silicon response."""
        time.sleep(3.0) # Increased to 3s to allow share propagation
        now = time.perf_counter()
        recent = [s for s in self.share_log if s[0] > (now - 5.0)]
        count = len(recent)
        
        jitter = [abs(s[0] - t_sent) for s in recent]
        features = {
            "timestamp": datetime.now().isoformat(),
            "share_count": count,
            "jitter_mean": np.mean(jitter) if count > 0 else 0,
            "jitter_std": np.std(jitter) if count > 1 else 0,
            "label": label # 1 for Golden, 0 for Loser
        }
        self.dataset.append(features)
        return features

    def save_dataset(self):
        df = pd.DataFrame(self.dataset)
        df.to_csv(self.config.DATASET_FILE, index=False)
        print(f"[DISK] Dataset saved to {self.config.DATASET_FILE} ({len(df)} samples)")

def fetch_historical_blocks(count=10):
    """Fetches real block data from a public explorer for Replay."""
    print(f"[WEB] Fetching {count} recent block headers...")
    blocks = []
    try:
        # Get latest block hash
        res = requests.get("https://blockchain.info/latestblock").json()
        curr_hash = res['hash']
        
        for _ in range(count):
            data = requests.get(f"https://blockchain.info/rawblock/{curr_hash}").json()
            # Extract header components (simplified for Stratum simulation)
            # Stratum expects: prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime
            # We will use the real winning nonce as a constant for the replay
            block_data = [
                data['prev_block'],
                "0"*70, "0"*20, # Simulated coinbase
                data['mrkl_root'], # Simplified merkle
                hex(data['ver'])[2:],
                hex(data['bits'])[2:],
                hex(data['time'])[2:],
                True
            ]
            blocks.append(block_data)
            curr_hash = data['prev_block']
            time.sleep(0.5) # API rate limit
    except Exception as e:
        print(f"[ERR] Web Fetch: {e}")
    return blocks

def main():
    config = Config()
    collector = TPFCollector(config)
    collector.start()
    
    print("="*60)
    print("   TPF TELEMETRY COLLECTOR: BLOCKCHAIN-INFORMED TRAINING")
    print("="*60)
    
    while not collector.authorized:
        print("[WAIT] Connect Miner...")
        time.sleep(2)

    # PHASE A: REPLAY HISTORICAL WINNERS (Golden Samples)
    historical_blocks = fetch_historical_blocks(config.REPLAY_BLOCKS)
    print(f"\n>>> PHASE A: REPLAYING {len(historical_blocks)} HISTORICAL WINNERS <<<")
    
    for i, block in enumerate(historical_blocks):
        print(f"   Replaying Block {i+1}...")
        # For winners, we use a very low difficulty to ensure the signature is clear
        jid, t_sent = collector.inject_job(0.0001, header_data=block)
        collector.capture_signature(t_sent, label=1)
        
    # PHASE B: COLLECT LOSERS (Negative Samples)
    print(f"\n>>> PHASE B: COLLECTING LOSER SIGNATURES ({config.LOSER_STEPS} steps) <<<")
    for i in range(config.LOSER_STEPS):
        # Higher random difficulty to simulate losers
        diff = 0.05 + (np.random.rand() * 0.1)
        jid, t_sent = collector.inject_job(diff)
        collector.capture_signature(t_sent, label=0)
        if i % 10 == 0:
            print(f"   Collected Losers: {i}/{config.LOSER_STEPS} | Shares: {len(collector.share_log)}")

    # FINALIZE
    collector.save_dataset()
    collector.running = False
    print("\n[DONE] Balanced Dataset ready for Deep Learning Training.")

if __name__ == "__main__":
    main()
