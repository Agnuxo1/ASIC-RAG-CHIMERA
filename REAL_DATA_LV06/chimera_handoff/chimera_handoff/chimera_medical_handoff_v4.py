#!/usr/bin/env python3
"""
ASIC-RAG-HEALTH: Validated Medical Record Sealing Experiment v4.0

BASADO EN: El código v2 del usuario (que FUNCIONA)
AÑADIDO: Verificación criptográfica REAL de shares

Cambios respecto a v2 del usuario:
1. Extrae nonce REAL del mining.submit
2. Intenta verificar SHA256d(header) < target
3. Guarda TODOS los datos para verificación independiente
4. Reporta honestamente si la verificación Python falla

Author: Francisco Angulo de Lafuente
Date: December 2025
"""

import socket
import threading
import json
import time
import os
import csv
import hashlib
import struct
import base64
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Crypto library
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    HAS_CRYPTO = True
except ImportError:
    print("WARNING: pycryptodome not installed. Using demo encryption.")
    HAS_CRYPTO = False

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    
    # Difficulty (tuned for ~1-10s response time on LV06)
    D_BASE = 0.08
    
    # Experiment
    DURATION_SEC = 300  # 5 minutes
    TIMEOUT_PER_SHARE = 60.0
    
    # Hardware specs
    LV06_POWER_WATTS = 9.0
    RTX3090_HASHRATE_MHS = 120.0
    RTX3090_POWER_WATTS = 350.0
    
    # Stratum
    EXTRANONCE1 = "08000002"
    EXTRANONCE2_SIZE = 4
    
    # Output
    OUTPUT_DIR = Path("chimera_v4_results")
    
    # Security
    MASTER_KEY = "HOSPITAL_ADDIS_GENERAL_KEY_2025"


# =============================================================================
# CRYPTOGRAPHIC UTILITIES
# =============================================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 (Bitcoin standard)"""
    return sha256(sha256(data))

def difficulty_to_target(difficulty: float) -> int:
    """Convert pool difficulty to target integer"""
    max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return int(max_target / difficulty)

def verify_hash_meets_target(hash_bytes: bytes, target: int) -> bool:
    """Check if hash (as little-endian int) is less than target"""
    hash_int = int.from_bytes(hash_bytes, byteorder='little')
    return hash_int < target


# =============================================================================
# ENCRYPTION (AES-256-GCM or Demo)
# =============================================================================

class CryptoEngine:
    @staticmethod
    def derive_key(passphrase: str) -> bytes:
        return hashlib.sha256(passphrase.encode()).digest()

    @staticmethod
    def encrypt(data: Dict, passphrase: str) -> str:
        if HAS_CRYPTO:
            key = CryptoEngine.derive_key(passphrase)
            nonce = get_random_bytes(12)
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            json_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            ciphertext, tag = cipher.encrypt_and_digest(json_bytes)
            payload = nonce + tag + ciphertext
            return base64.b64encode(payload).decode('utf-8')
        else:
            # Demo fallback
            return base64.b64encode(json.dumps(data).encode()).decode()


# =============================================================================
# MEDICAL DATA
# =============================================================================

@dataclass
class MedicalRecord:
    record_id: str
    patient_id: str
    timestamp: float
    data: Dict
    encrypted_blob: str
    sha256_tag: str

class MerkleTree:
    def __init__(self, records: List[MedicalRecord]):
        self.leaves = [bytes.fromhex(r.sha256_tag) for r in records]
        self.root = self._compute_root()
        
    def _compute_root(self) -> bytes:
        if not self.leaves:
            return b'\x00' * 32
        level = self.leaves[:]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                next_level.append(sha256d(left + right))
            level = next_level
        return level[0]
    
    def get_root_hex(self) -> str:
        return self.root.hex()

class MedicalRecordGenerator:
    NAMES = ["Abebe Bikila", "Derartu Tulu", "Haile Gebrselassie", "Tirunesh Dibaba"]
    
    def __init__(self):
        self.counter = 0
    
    def generate_batch(self, count: int = 5) -> List[MedicalRecord]:
        records = []
        patient_name = self.NAMES[self.counter % len(self.NAMES)]
        patient_id = f"ETH-PAT-{self.counter:04d}"
        self.counter += 1
        
        for i in range(count):
            rid = f"REC-{self.counter}-{i}"
            raw_data = {
                "name": patient_name,
                "bp": f"{120 + i}/{80 + i}",
                "diagnosis": "Malaria Screening",
                "result": "Negative"
            }
            
            enc_blob = CryptoEngine.encrypt(raw_data, Config.MASTER_KEY)
            tag = hashlib.sha256(f"{rid}:{enc_blob}".encode()).hexdigest()
            
            records.append(MedicalRecord(
                rid, patient_id, time.time(), raw_data, enc_blob, tag
            ))
        return records


# =============================================================================
# VERIFIED EVENT STRUCTURE
# =============================================================================

@dataclass
class VerifiedEvent:
    """Complete data for independent verification"""
    # Timing
    timestamp: float
    job_sent_at: float
    share_received_at: float
    latency_s: float
    
    # Medical binding
    merkle_root: str
    records_sealed: int
    
    # ASIC submission (extracted from mining.submit)
    worker_name: str
    job_id: str
    extranonce2: str
    ntime: str
    nonce: str
    
    # Verification
    difficulty: float
    target_hex: str
    block_header_hex: str
    block_hash_hex: str
    hash_meets_target: bool
    verification_method: str  # "python" or "miner_trusted"


# =============================================================================
# STRATUM BRIDGE (Based on user's working code)
# =============================================================================

class VerifiedBridge(threading.Thread):
    def __init__(self, out_dir: Path):
        super().__init__(daemon=True)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Socket setup (exactly like user's working code)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((Config.HOST, Config.PORT))
        self.sock.listen(5)
        
        # State
        self.conn = None
        self.authorized = False
        self.running = True
        
        # Data collection
        self.events: List[VerifiedEvent] = []
        self.generator = MedicalRecordGenerator()
        
        # Share notification
        self.share_received = threading.Event()
        self.last_submit_params = None
        self.last_submit_time = None
        
        # Current job tracking
        self.current_job = None

    def run(self):
        print(f"[BRIDGE] Listening on {Config.HOST}:{Config.PORT}")
        while self.running:
            try:
                conn, addr = self.sock.accept()
                print(f"[BRIDGE] ASIC Connected from {addr[0]}")
                self.conn = conn
                self._handle_client()
            except Exception as e:
                if self.running:
                    print(f"[BRIDGE] Accept error: {e}")

    def _handle_client(self):
        """Single loop for all messages (like user's working code)"""
        buffer = ""
        self.conn.settimeout(0.5)  # Short timeout, not non-blocking!
        
        while self.running:
            try:
                data = self.conn.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break
                buffer += data
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            self._process_msg(msg)
                        except json.JSONDecodeError:
                            pass
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[BRIDGE] Connection error: {e}")
                break
        
        if self.conn:
            self.conn.close()
            self.conn = None

    def _process_msg(self, msg: Dict):
        """Process Stratum messages"""
        mid = msg.get('id')
        method = msg.get('method')
        
        if method == 'mining.subscribe':
            res = [
                [["mining.set_difficulty", "1"], ["mining.notify", "1"]], 
                Config.EXTRANONCE1, 
                Config.EXTRANONCE2_SIZE
            ]
            self._send({"id": mid, "result": res, "error": None})
            print("[BRIDGE] Subscribed")
            
        elif method == 'mining.authorize':
            self._send({"id": mid, "result": True, "error": None})
            self.authorized = True
            print("[BRIDGE] Authorized - Ready for jobs")
            
        elif method == 'mining.submit':
            # CRITICAL: Extract ALL parameters
            params = msg.get('params', [])
            if len(params) >= 5:
                self.last_submit_params = {
                    'worker_name': params[0],
                    'job_id': params[1],
                    'extranonce2': params[2],
                    'ntime': params[3],
                    'nonce': params[4]
                }
                self.last_submit_time = time.time()
                
                # Always accept to keep miner happy
                self._send({"id": mid, "result": True, "error": None})
                self.share_received.set()
                
                print(f"[BRIDGE] Share received: nonce={params[4]}")
            else:
                self._send({"id": mid, "result": False, "error": [20, "Bad params", None]})

    def _send(self, data: Dict):
        """Send JSON message to miner"""
        if self.conn:
            try:
                self.conn.sendall((json.dumps(data) + '\n').encode())
            except:
                pass

    def seal_medical_batch(self, timeout: float = None) -> Optional[VerifiedEvent]:
        """Seal a batch of medical records with verified PoW"""
        if not self.authorized:
            return None
            
        timeout = timeout or Config.TIMEOUT_PER_SHARE
        
        # 1. Generate medical records
        batch = self.generator.generate_batch(5)
        tree = MerkleTree(batch)
        merkle_root = tree.get_root_hex()
        
        # 2. Set difficulty
        self._send({
            "id": None, 
            "method": "mining.set_difficulty", 
            "params": [Config.D_BASE]
        })
        
        # 3. Create and send job
        job_id = f"med_{int(time.time() * 1000)}"
        ntime_hex = hex(int(time.time()))[2:]
        
        # Job structure (like user's working code)
        self.current_job = {
            "job_id": job_id,
            "prevhash": merkle_root,
            "coinb1": "01" * 32,
            "coinb2": "00000000",
            "merkle_branches": [],
            "version": "20000000",
            "nbits": "1f00ffff",
            "ntime": ntime_hex,
            "sent_at": time.time()
        }
        
        # mining.notify params
        params = [
            job_id,
            merkle_root,
            self.current_job["coinb1"],
            self.current_job["coinb2"],
            [],  # merkle branches
            self.current_job["version"],
            self.current_job["nbits"],
            ntime_hex,
            True  # clean_jobs
        ]
        
        self.share_received.clear()
        self._send({"id": None, "method": "mining.notify", "params": params})
        
        print(f"[SEAL] Job {job_id} sent, merkle={merkle_root[:16]}...")
        
        # 4. Wait for share
        if not self.share_received.wait(timeout):
            print("[SEAL] Timeout")
            return None
        
        submit = self.last_submit_params
        submit_time = self.last_submit_time
        
        if not submit or submit['job_id'] != job_id:
            print(f"[SEAL] Job mismatch")
            return None
        
        # 5. Attempt cryptographic verification
        event = self._verify_and_record(
            job=self.current_job,
            submit=submit,
            submit_time=submit_time,
            merkle_root=merkle_root,
            records_count=len(batch)
        )
        
        if event:
            self.events.append(event)
            
        return event

    def _verify_and_record(
        self,
        job: Dict,
        submit: Dict,
        submit_time: float,
        merkle_root: str,
        records_count: int
    ) -> VerifiedEvent:
        """
        Attempt to verify the share cryptographically.
        Records all data regardless of verification success.
        """
        latency = submit_time - job['sent_at']
        target = difficulty_to_target(Config.D_BASE)
        target_hex = format(target, '064x')
        
        # Try to verify with Python
        verified_py = False
        block_header_hex = ""
        block_hash_hex = ""
        
        try:
            # Construct coinbase
            coinbase_hex = (
                job['coinb1'] + 
                Config.EXTRANONCE1 + 
                submit['extranonce2'] + 
                job['coinb2']
            )
            coinbase = bytes.fromhex(coinbase_hex)
            coinbase_hash = sha256d(coinbase)
            
            # Construct block header (80 bytes)
            # Version: 4 bytes LE
            version = struct.pack('<I', int(job['version'], 16))
            
            # Prevhash: 32 bytes (reversed from hex)
            prevhash = bytes.fromhex(job['prevhash'])[::-1]
            
            # Merkle root of block (= coinbase hash for no branches)
            block_merkle = coinbase_hash
            
            # ntime: 4 bytes LE
            ntime = struct.pack('<I', int(submit['ntime'], 16))
            
            # nbits: 4 bytes LE
            nbits = struct.pack('<I', int(job['nbits'], 16))
            
            # nonce: 4 bytes LE
            nonce = struct.pack('<I', int(submit['nonce'], 16))
            
            header = version + prevhash + block_merkle + ntime + nbits + nonce
            block_header_hex = header.hex()
            
            # Hash the header
            block_hash = sha256d(header)
            block_hash_hex = block_hash[::-1].hex()  # Reverse for display
            
            # Verify
            verified_py = verify_hash_meets_target(block_hash, target)
            
        except Exception as e:
            print(f"[VERIFY] Python verification error: {e}")
        
        # Determine verification status
        if verified_py:
            method = "python_verified"
            status = "✓ VERIFIED (Python)"
        else:
            method = "miner_trusted"
            status = "⚠ Miner-trusted (Python verification failed)"
        
        print(f"[VERIFY] {status} | nonce={submit['nonce']} | latency={latency:.2f}s")
        
        # Record everything for independent verification
        event = VerifiedEvent(
            timestamp=submit_time,
            job_sent_at=job['sent_at'],
            share_received_at=submit_time,
            latency_s=latency,
            merkle_root=merkle_root,
            records_sealed=records_count,
            worker_name=submit['worker_name'],
            job_id=submit['job_id'],
            extranonce2=submit['extranonce2'],
            ntime=submit['ntime'],
            nonce=submit['nonce'],
            difficulty=Config.D_BASE,
            target_hex=target_hex,
            block_header_hex=block_header_hex,
            block_hash_hex=block_hash_hex,
            hash_meets_target=verified_py,  # Honest reporting
            verification_method=method
        )
        
        return event

    def save_results(self) -> Dict:
        """Save all results and compute statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save events as JSON (complete data)
        events_file = self.out_dir / f"verified_events_{timestamp}.json"
        with open(events_file, 'w') as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
        print(f"[SAVE] Events: {events_file}")
        
        # 2. Save as CSV for analysis
        csv_file = self.out_dir / f"events_{timestamp}.csv"
        if self.events:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.events[0]).keys())
                writer.writeheader()
                for e in self.events:
                    writer.writerow(asdict(e))
        print(f"[SAVE] CSV: {csv_file}")
        
        # 3. Calculate statistics
        if not self.events:
            return {}
        
        latencies = [e.latency_s for e in self.events]
        verified_py = len([e for e in self.events if e.verification_method == "python_verified"])
        
        # Hashrate calculation
        total_shares = len(self.events)
        total_time = self.events[-1].share_received_at - self.events[0].job_sent_at
        total_hashes = total_shares * Config.D_BASE * (2**32)
        
        if total_time > 0:
            hashrate_mhs = (total_hashes / total_time) / 1e6
        else:
            hashrate_mhs = 0
        
        efficiency_mhw = hashrate_mhs / Config.LV06_POWER_WATTS
        gpu_efficiency = Config.RTX3090_HASHRATE_MHS / Config.RTX3090_POWER_WATTS
        vs_gpu = efficiency_mhw / gpu_efficiency if gpu_efficiency > 0 else 0
        
        stats = {
            "experiment_date": datetime.now().isoformat(),
            "duration_s": total_time,
            "total_shares": total_shares,
            "verified_by_python": verified_py,
            "verified_by_miner_trust": total_shares - verified_py,
            "python_verification_rate": verified_py / total_shares if total_shares > 0 else 0,
            "total_records_sealed": sum(e.records_sealed for e in self.events),
            "avg_latency_s": sum(latencies) / len(latencies),
            "min_latency_s": min(latencies),
            "max_latency_s": max(latencies),
            "hashrate_mhs": hashrate_mhs,
            "efficiency_mhw": efficiency_mhw,
            "vs_rtx3090_efficiency": vs_gpu,
            "power_watts": Config.LV06_POWER_WATTS
        }
        
        # Save stats
        stats_file = self.out_dir / f"statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[SAVE] Stats: {stats_file}")
        
        # Print summary
        self._print_summary(stats)
        
        return stats

    def _print_summary(self, stats: Dict):
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        print(f"Duration:              {stats['duration_s']:.1f}s")
        print(f"Total shares:          {stats['total_shares']}")
        print(f"Python-verified:       {stats['verified_by_python']} ({stats['python_verification_rate']*100:.0f}%)")
        print(f"Miner-trusted:         {stats['verified_by_miner_trust']}")
        print(f"Records sealed:        {stats['total_records_sealed']}")
        print(f"Avg latency:           {stats['avg_latency_s']:.2f}s")
        print(f"Hashrate:              {stats['hashrate_mhs']:.2f} MH/s")
        print(f"Efficiency:            {stats['efficiency_mhw']:.2f} MH/W")
        print(f"vs RTX 3090:           {stats['vs_rtx3090_efficiency']:.2f}x")
        print("=" * 60)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except:
            pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ASIC-RAG-HEALTH v4.0 - Verified Medical Sealing")
    print("=" * 60)
    print(f"Encryption: {'AES-256-GCM' if HAS_CRYPTO else 'Demo (base64)'}")
    print(f"Duration:   {Config.DURATION_SEC}s")
    print(f"Difficulty: {Config.D_BASE}")
    print(f"Output:     {Config.OUTPUT_DIR}")
    print()
    
    bridge = VerifiedBridge(Config.OUTPUT_DIR)
    bridge.start()
    
    print(f"Waiting for LV06 connection on port {Config.PORT}...")
    print(f"Point your miner to: {socket.gethostbyname(socket.gethostname())}:{Config.PORT}")
    print()
    
    # Wait for connection
    try:
        while not bridge.authorized:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
        return
    
    print("\nStarting experiment...\n")
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < Config.DURATION_SEC:
            event = bridge.seal_medical_batch()
            
            if event:
                elapsed = time.time() - start_time
                remaining = Config.DURATION_SEC - elapsed
                print(f"[PROGRESS] {elapsed:.0f}s elapsed, {remaining:.0f}s remaining, "
                      f"{len(bridge.events)} shares")
    except KeyboardInterrupt:
        print("\nStopping...")
    
    print("\nSaving results...")
    bridge.save_results()
    bridge.stop()
    
    print(f"\nResults saved to {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
