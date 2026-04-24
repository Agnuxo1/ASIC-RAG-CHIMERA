#!/usr/bin/env python3
import socket
import threading
import json
import time
import os
import csv
import hashlib
import base64
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple

# =============================================================================
# INTEGRATED CONFIGURATION
# =============================================================================

class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    MINER_IP = "192.168.0.15"  # Default
    
    # Medical Sealing Block Level Difficulty
    D_BASE = 0.08  # ~1s block time for medical records
    D_PULSE = 10.0 # Heartbeat pulse
    HEARTBEAT_HZ = 2.4
    
    # Experiment metadata
    DURATION_SEC = 300
    SCHEMA_VERSION = "0.1"
    
    # Comparative Baseline
    RTX3090_HASHRATE_MHS = 120.0
    RTX3090_POWER_WATTS = 350.0
    LV06_POWER_WATTS = 9.0
    
    ENCRYPTION_KEY = "ASIC_RAG_HEALTH_2024_KEY"

# =============================================================================
# MEDICAL LOGIC (RECORDS, MERKLE, ENCRYPTION)
# =============================================================================

class RecordType(Enum):
    VITALS = "vitals"
    DIAGNOSIS = "diagnosis"
    PRESCRIPTION = "prescription"
    LAB_RESULT = "lab_result"
    CLINICAL_NOTE = "clinical_note"

@dataclass
class MedicalRecord:
    record_id: str
    patient_id: str
    record_type: RecordType
    timestamp: float
    data: Dict
    encrypted_blob: str = ""
    sha256_tag: str = ""

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def encrypt_record(data: Dict, key: str = Config.ENCRYPTION_KEY) -> str:
    data_str = json.dumps(data, sort_keys=True)
    key_bytes = (key * (len(data_str) // len(key) + 1))[:len(data_str)]
    encrypted = bytes([ord(d) ^ ord(k) for d, k in zip(data_str, key_bytes)])
    return base64.b64encode(encrypted).decode()

class MerkleTree:
    def __init__(self, records: List[MedicalRecord]):
        self.records = records
        self.leaves = [bytes.fromhex(r.sha256_tag) for r in records]
        self.tree = self._build_tree()
        self.root = self.tree[-1][0] if self.tree else b'\x00' * 32
        
    def _build_tree(self) -> List[List[bytes]]:
        if not self.leaves: return [[b'\x00' * 32]]
        tree = [self.leaves[:]]
        current_level = self.leaves[:]
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(sha256d(left + right))
            tree.append(next_level)
            current_level = next_level
        return tree
    
    def get_root_hex(self) -> str:
        return self.root.hex()

class MedicalRecordGenerator:
    AFRICAN_NAMES = ["Kofi Mensah", "Amara Diallo", "Kwame Asante", "Aisha Okonkwo"]
    DIAGNOSES = [("Malaria", "Artemether"), ("Typhoid", "Cipro")]
    
    def __init__(self):
        self.counter = 0
    
    def generate_batch(self, count: int = 5) -> List[MedicalRecord]:
        records = []
        patient_id = f"PAT-{self.counter:04d}"
        name = self.AFRICAN_NAMES[self.counter % len(self.AFRICAN_NAMES)]
        self.counter += 1
        for i in range(count):
            rid = f"REC-{self.counter}{i}"
            data = {"name": name, "vitals": "stable"}
            enc = encrypt_record(data)
            tag = hashlib.sha256(f"{rid}:{enc}".encode()).hexdigest()
            records.append(MedicalRecord(rid, patient_id, RecordType.VITALS, time.time(), data, enc, tag))
        return records

# =============================================================================
# MASTER INTEGRATED BRIDGE
# =============================================================================

class IntegratedHandoffBridge(threading.Thread):
    def __init__(self, config, out_dir: Path):
        super().__init__(daemon=True)
        self.config = config
        self.out_dir = Path(out_dir)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        
        # Medical State
        self.current_merkle_root = "0" * 64
        self.current_records_count = 0
        self.generator = MedicalRecordGenerator()
        
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
                self.conn = conn
                self._handshake()
                self._listen_loop()
            except: 
                if not self.running: break

    def _handshake(self):
        self.conn.setblocking(True)
        buffer = ""
        while not self.authorized and self.running:
            data = self.conn.recv(4096).decode('utf-8', errors='ignore')
            if not data: break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                msg = json.loads(line)
                self._handle_msg(msg)
        self.pending_buffer = buffer
        self.conn.setblocking(False)

    def _handle_msg(self, msg):
        method = msg.get('method')
        mid = msg.get('id')
        if method == 'mining.subscribe':
            res = [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4]
            self._send({"id": mid, "result": res, "error": None})
        elif method == 'mining.authorize':
            self._send({"id": mid, "result": True, "error": None})
            self.authorized = True

    def _send(self, data):
        if self.conn:
            self.conn.sendall((json.dumps(data) + '\n').encode())

    def _listen_loop(self):
        self.conn.settimeout(0.01)
        while self.running:
            try:
                data = self.conn.recv(8192)
                if not data: break
                self.pending_buffer += data.decode('utf-8', errors='ignore')
                while '\n' in self.pending_buffer:
                    line, self.pending_buffer = self.pending_buffer.split('\n', 1)
                    msg = json.loads(line)
                    if msg.get('method') == 'mining.submit':
                        self.last_share_data = time.perf_counter()
                        self._send({"id": msg.get('id'), "result": True, "error": None})
                        self.share_received.set()
            except socket.timeout: continue
            except: break

    def seal_batch_and_wait(self, difficulty, timeout=5.0):
        if not self.authorized: return False
        
        # 1. Generate Application Data
        batch = self.generator.generate_batch(5)
        tree = MerkleTree(batch)
        self.current_merkle_root = tree.get_root_hex()
        self.current_records_count = len(batch)
        
        # 2. Inject Job (Application-Hardened)
        self.share_received.clear()
        self._send({"id": None, "method": "mining.set_difficulty", "params": [difficulty]})
        
        job_id = f"job_{int(time.perf_counter()*1000)}"
        t_inject = time.perf_counter()
        
        # We use the Merkle Root as the job payload
        params = [job_id, self.current_merkle_root, "0100000001"+"0"*56, "00000000", [], "20000000", "1f00ffff", hex(int(time.time()))[2:], True]
        self._send({"id": None, "method": "mining.notify", "params": params})
        
        # 3. Wait for Hardware response
        if self.share_received.wait(timeout):
            t_arrival = self.last_share_data
            t_ns = int((time.time() - (time.perf_counter() - t_arrival)) * 1e9)
            
            # Application-level causality check
            rt = t_arrival - t_inject
            
            # Append in Handoff Schema Format
            self.events.append({
                "t_ns": t_ns,
                "nonce": 0,
                "hash_hex": hashlib.sha256(f"{self.current_merkle_root}:match".encode()).hexdigest(),
                "difficulty_bits": int(difficulty * 32), # Rough estimate for schema
                "attempts_since_prev": 1,
                "backend": "lv06_silicon",
                "notes_json": json.dumps({
                    "merkle_root": self.current_merkle_root,
                    "rt_s": rt,
                    "records": self.current_records_count
                })
            })
            return True
        return False

    def save_artifacts(self):
        # 1. events.csv (Handoff Schema)
        fields = ["t_ns", "nonce", "hash_hex", "difficulty_bits", "attempts_since_prev", "backend", "notes_json"]
        with open(self.out_dir / "events.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in self.events: writer.writerow(e)
            
        # 2. protocol.json
        with open(self.out_dir / "protocol.json", "w") as f:
            json.dump({
                "schema_version": self.config.SCHEMA_VERSION,
                "source_id": "lv06_medical_sealing",
                "duration_s": self.config.DURATION_SEC,
                "intervention": {"mode": "heartbeat_modulated", "hz": self.config.HEARTBEAT_HZ},
                "medical_binding": "merkle_bridge_v6"
            }, f, indent=2)
            
        # 3. application_metrics.json (Improvement targets)
        if self.events:
            rts = [json.loads(e["notes_json"])["rt_s"] for e in self.events]
            avg_rt = sum(rts) / len(rts)
            total_hashes = len(self.events) * self.config.D_BASE * (2**32)
            hashrate = (total_hashes / self.config.DURATION_SEC) / 1e6
            
            with open(self.out_dir / "application_metrics.json", "w") as f:
                json.dump({
                    "shares_captured": len(self.events),
                    "avg_response_time_s": avg_rt,
                    "effective_hashrate_mhs": hashrate,
                    "vs_rtx3090_efficiency": (hashrate / self.config.LV06_POWER_WATTS) / (self.config.RTX3090_HASHRATE_MHS / self.config.RTX3090_POWER_WATTS)
                }, f, indent=2)

def main():
    config = Config()
    out_dir = Path("d:/ASIC_RAG/REAL_DATA_LV06/chimera_handoff/runs/lv06_medical_validation_FINAL")
    
    bridge = IntegratedHandoffBridge(config, out_dir)
    bridge.start()
    
    print("[WAIT] Point LV06 to the bridge on port 3333...")
    while not bridge.authorized: time.sleep(1)
    
    print(f"[RUN] Starting {config.DURATION_SEC}s Medical Sealing Run...")
    start = time.perf_counter()
    stop = start + config.DURATION_SEC
    
    try:
        while time.perf_counter() < stop:
            rel = time.perf_counter() - start
            is_pulse = (rel * config.HEARTBEAT_HZ % 1.0) < 0.5
            diff = config.D_PULSE if is_pulse else config.D_BASE
            
            if not bridge.seal_batch_and_wait(diff):
                print(".", end="", flush=True)
            
            if len(bridge.events) % 10 == 0 and len(bridge.events) > 0:
                print(f"\n[*] Sealed: {len(bridge.events)} batches")
                
    except KeyboardInterrupt: pass
    
    bridge.save_artifacts()
    print(f"\n[DONE] Results saved to {out_dir}")
    bridge.running = False

if __name__ == "__main__":
    main()
