#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTIC BRIDGE: Packet-Level Ingestion (v6)
================================================================================
Scientific Objective:
Exhaustive debug of the Stratum handshake and share submission.
Observes why shares are reported 'Accepted' in UI but not captured in bridge.
================================================================================
"""

import socket
import threading
import json
import time
import os
import csv
import sys
import struct
from pathlib import Path
from datetime import datetime

class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    MINER_IP = "192.168.0.15"
    
    DURATION_SEC = 240
    HEARTBEAT_HZ = 2.4
    
    D_BASE = 0.005
    D_PULSE = 10.0 # Pulsing for visibility
    
    RUN_ID = f"lv06_sovereignty_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUT_DIR = Path(f"d:/ASIC_RAG/REAL_DATA_LV06/chimera_handoff/runs/{RUN_ID}")

class DiagnosticBridge(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(5)
        
        self.conn = None
        self.authorized = False
        self.running = True
        self.events = []
        self.share_received = threading.Event()
        self.last_share_data = None
        self.pending_buffer = ""

    def run(self):
        print(f"[BRIDGE] Diagnostic Listener on {self.config.PORT}...", flush=True)
        try:
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    print(f"[BRIDGE] Client connected: {addr}", flush=True)
                    # We only support one active miner for this experiment
                    self.conn = conn
                    self._handshake()
                    self._listen_loop()
                except Exception as e:
                    print(f"[ERR] Accept/Loop: {e}", flush=True)
                    if not self.running: break
        except Exception as e:
            print(f"[ERR] Server: {e}", flush=True)

    def _handshake(self):
        self.conn.setblocking(True)
        self.conn.settimeout(10.0)
        buffer = ""
        while not self.authorized and self.running:
            try:
                data = self.conn.recv(4096).decode('utf-8', errors='ignore')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip(): continue
                    msg = json.loads(line)
                    print(f" [RECV] {msg.get('method') or 'res'}: {line[:120]}...", flush=True)
                    self._handle_msg(msg)
            except socket.timeout: 
                print("[WARN] Handshake timeout.", flush=True)
                break
        self.pending_buffer = buffer
        self.conn.setblocking(False)

    def _listen_loop(self):
        self.conn.settimeout(0.01)
        while self.running:
            try:
                data = self.conn.recv(8192)
                if not data: break
                self.pending_buffer += data.decode('utf-8', errors='ignore')
                while '\n' in self.pending_buffer:
                    line, self.pending_buffer = self.pending_buffer.split('\n', 1)
                    if not line.strip(): continue
                    try:
                        self._process_msg(json.loads(line), line)
                    except Exception as e:
                        print(f" [ERR] JSON Parse: {e} | Line: {line}", flush=True)
            except socket.timeout: continue
            except: break

    def _handle_msg(self, msg):
        method = msg.get('method')
        mid = msg.get('id')
        if method == 'mining.subscribe':
            res = [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4]
            self._send({"id": mid, "result": res, "error": None})
        elif method == 'mining.authorize':
            self._send({"id": mid, "result": True, "error": None})
            self.authorized = True
            print("[BRIDGE] Authorized Miner.", flush=True)

    def _process_msg(self, msg, raw_line):
        method = msg.get('method')
        if method == 'mining.submit':
            t_now = time.perf_counter()
            print(f" [SHARE] Match! ID={msg.get('id')}", flush=True)
            self._send({"id": msg.get('id'), "result": True, "error": None})
            self.last_share_data = t_now
            self.share_received.set()
        else:
            # Log any other messages to stay informed
            print(f" [MSG] {method or 'response'}: {raw_line[:80]}...", flush=True)

    def _send(self, data):
        if self.conn:
            try:
                self.conn.sendall((json.dumps(data) + '\n').encode())
            except Exception as e:
                print(f" [ERR] Send: {e}", flush=True)

    def inject_and_wait(self, diff, timeout=5.0):
        if not self.authorized: return False
        self.share_received.clear()
        
        # 1. Difficulty
        self._send({"id": None, "method": "mining.set_difficulty", "params": [diff]})
        
        # 2. Job
        job_id = f"job_{int(time.perf_counter()*1000)}"
        t_inject = time.perf_counter()
        # Using a slightly more 'realistic' coinbase to avoid rejection
        params = [job_id, "0"*64, "0100000001"+"0"*56, "00000000", [], "20000000", "1f00ffff", hex(int(time.time()))[2:], True]
        self._send({"id": None, "method": "mining.notify", "params": params})
        
        # 3. Wait
        if self.share_received.wait(timeout):
            t_arrival = self.last_share_data
            t_ns = int((time.time() - (time.perf_counter() - t_arrival)) * 1e9)
            self.events.append({
                "t_ns": t_ns,
                "nonce": 0,
                "hash_hex": "0"*64,
                "backend": "lv06_silicon",
                "notes_json": json.dumps({"rt": t_arrival - t_inject, "jid": job_id})
            })
            return True
        return False

def main():
    config = Config()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bridge = DiagnosticBridge(config)
    bridge.start()
    
    # Wait for miner
    while not bridge.authorized:
        time.sleep(1)
        
    print(f"[RUN] Starting {config.DURATION_SEC}s Diagnostic Run...", flush=True)
    start_time = time.perf_counter()
    stop_time = start_time + config.DURATION_SEC
    period = 1.0 / config.HEARTBEAT_HZ
    
    try:
        while time.perf_counter() < stop_time:
            now = time.perf_counter()
            rel_now = now - start_time
            
            is_pulse = (rel_now * config.HEARTBEAT_HZ % 1.0) < 0.5
            current_diff = config.D_PULSE if is_pulse else config.D_BASE
            
            if not bridge.inject_and_wait(current_diff, timeout=2.0):
                # We print a dot for every timeout to monitor health
                print(".", end="", flush=True)
            
            if len(bridge.events) % 10 == 0 and len(bridge.events) > 0:
                print(f"\n[*] Collected: {len(bridge.events)}", flush=True)
                
    except KeyboardInterrupt: pass
        
    print(f"\n[DONE] Captured {len(bridge.events)} events.", flush=True)
    
    # EXPORT
    with open(config.OUT_DIR / "events.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_ns", "nonce", "hash_hex", "backend", "notes_json"])
        writer.writeheader()
        for e in bridge.events: writer.writerow(e)
            
    with open(config.OUT_DIR / "protocol.json", "w") as f:
        json.dump({
            "schema_version": "0.1",
            "source_id": "lv06_silicon_reservoir",
            "duration_s": config.DURATION_SEC,
            "intervention": {"mode": "diagnostic_v6", "hz": config.HEARTBEAT_HZ}
        }, f, indent=2)
        
    print(f"[DISK] Saved to: {config.OUT_DIR}")
    bridge.running = False

if __name__ == "__main__":
    main()
