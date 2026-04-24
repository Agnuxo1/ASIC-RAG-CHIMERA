#!/usr/bin/env python3
"""
ASIC-RAG-HEALTH: Cryptographically Verified Medical Record Sealing v7.0

FIX CRÍTICO v7: Corrección de target_to_nbits

El problema en v6 era que target_to_nbits producía nBits incorrectos:
- V6 enviaba nBits=0x1d0000c7, que da target=0x00c7... (muy alto)
- El target correcto para diff=0.08 es 0x0c7ff38...
- nBits correcto debe ser 0x1d0c7ff3

Esto causaba que el ASIC minara a ~4100x menos dificultad de lo esperado,
por lo que sus shares nunca pasaban la verificación Python.

Author: Francisco Angulo de Lafuente
Date: December 2025
License: MIT
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
from typing import List, Dict, Optional, Tuple
import traceback

# =============================================================================
# OPTIONAL CRYPTO
# =============================================================================

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    HAS_CRYPTO = True
except ImportError:
    print("[WARN] pycryptodome not installed. Using demo encryption.")
    HAS_CRYPTO = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    HOST = "0.0.0.0"
    PORT = 3333
    
    # Difficulty for shares - using miner's suggested value
    # The miner suggests 1000, which may be the optimal range
    # At diff=1000, expect ~1 share per 30s at 32 MH/s
    DIFFICULTY = 1000
    
    # Experiment
    DURATION_SEC = 300
    TIMEOUT_PER_SHARE = 120.0
    
    # Hardware
    LV06_POWER_WATTS = 9.0
    RTX3090_HASHRATE_MHS = 120.0
    RTX3090_POWER_WATTS = 350.0
    
    # Stratum
    EXTRANONCE1 = "deadbeef"
    EXTRANONCE2_SIZE = 4
    
    # Version for mining.notify (will be modified by ASIC via version-rolling)
    BLOCK_VERSION = 0x20000000
    
    # Version rolling mask (which bits the ASIC can modify)
    VERSION_ROLLING_MASK = 0x1fffe000  # Standard BIP320 mask
    
    # Output
    OUTPUT_DIR = Path("chimera_v7_verified")
    
    # Security
    MASTER_KEY = "HOSPITAL_ADDIS_GENERAL_KEY_2025"
    
    # Debug
    DEBUG = True


# =============================================================================
# CRYPTO PRIMITIVES
# =============================================================================

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256d(data: bytes) -> bytes:
    return sha256(sha256(data))

def reverse_bytes(data: bytes) -> bytes:
    return data[::-1]

def hex_to_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)

def bytes_to_hex(data: bytes) -> str:
    return data.hex()

def uint32_le(value: int) -> bytes:
    return struct.pack('<I', value)

def uint64_le(value: int) -> bytes:
    return struct.pack('<Q', value)


# =============================================================================
# TARGET/DIFFICULTY
# =============================================================================

def difficulty_to_target(difficulty: float) -> int:
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return int(diff1_target / difficulty)

def target_to_nbits(target: int) -> int:
    """
    Convert a target integer to compact nBits format.
    
    nBits format: 
    - High byte = exponent (size in bytes)
    - Low 3 bytes = mantissa (first 3 significant bytes)
    
    Target = mantissa * 2^(8 * (exponent - 3))
    """
    if target == 0:
        return 0
    
    # Convert target to bytes (big-endian)
    target_bytes = target.to_bytes((target.bit_length() + 7) // 8, 'big')
    
    # Size is the number of bytes
    size = len(target_bytes)
    
    # Mantissa is first 3 bytes (or less if target is small)
    if size >= 3:
        mantissa = int.from_bytes(target_bytes[:3], 'big')
    else:
        mantissa = int.from_bytes(target_bytes, 'big') << (8 * (3 - size))
    
    # If high bit of mantissa is set, we need to adjust
    # (because mantissa must be positive in the protocol)
    if mantissa & 0x800000:
        mantissa >>= 8
        size += 1
    
    # nBits = (size << 24) | mantissa
    return (size << 24) | (mantissa & 0x00FFFFFF)

def check_hash_meets_target(block_hash: bytes, target: int) -> bool:
    hash_int = int.from_bytes(block_hash, byteorder='little')
    return hash_int < target


# =============================================================================
# STRATUM HELPERS
# =============================================================================

def swap_endianness_32bit_words(hex_str: str) -> str:
    """Swap each 4-byte word's endianness (Stratum prevhash format)"""
    if len(hex_str) % 8 != 0:
        hex_str = hex_str.zfill((len(hex_str) // 8 + 1) * 8)
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        swapped = "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
        result += swapped
    return result

def parse_prevhash_from_stratum(prevhash_hex: str) -> bytes:
    """Convert Stratum prevhash to internal byte order"""
    unswapped = swap_endianness_32bit_words(prevhash_hex)
    return hex_to_bytes(unswapped)


# =============================================================================
# COINBASE CONSTRUCTION
# =============================================================================

def create_coinbase_parts() -> Tuple[str, str]:
    """
    Create valid Bitcoin coinbase transaction parts.
    
    Returns (coinb1_hex, coinb2_hex)
    """
    # Version
    version = uint32_le(1)
    
    # Input count
    input_count = bytes([1])
    
    # Previous tx (null for coinbase)
    prev_tx = bytes(32)
    
    # Previous index (0xffffffff for coinbase)
    prev_index = bytes.fromhex('ffffffff')
    
    # Script sig components
    # Block height encoding (BIP34)
    height = 1
    height_bytes = bytes([1, height & 0xff])
    
    # Pool tag
    prefix_data = b'CHIMERA/'
    suffix_data = b'/MED/'
    
    # Script sig = height + prefix + [extranonce1 + extranonce2 placeholder] + suffix
    # Total: 2 + 8 + 8 + 5 = 23 bytes
    script_prefix = height_bytes + prefix_data
    script_suffix = suffix_data
    
    # Total script length
    script_len = len(script_prefix) + 8 + len(script_suffix)
    
    # Sequence
    sequence = bytes.fromhex('ffffffff')
    
    # Output count
    output_count = bytes([1])
    
    # Output value (50 BTC)
    output_value = uint64_le(50 * 100000000)
    
    # Output script (OP_RETURN with marker)
    # OP_RETURN + push 16 bytes + "ASIC-RAG-HEALTH!"
    marker = b'ASIC-RAG-HEALTH!'
    output_script = bytes([0x6a, len(marker)]) + marker
    output_script_len = bytes([len(output_script)])
    
    # Locktime
    locktime = bytes(4)
    
    # Build coinb1
    coinb1 = (
        version +
        input_count +
        prev_tx +
        prev_index +
        bytes([script_len]) +
        script_prefix
    )
    
    # Build coinb2
    coinb2 = (
        script_suffix +
        sequence +
        output_count +
        output_value +
        output_script_len +
        output_script +
        locktime
    )
    
    return (bytes_to_hex(coinb1), bytes_to_hex(coinb2))


def build_coinbase(coinb1: str, extranonce1: str, extranonce2: str, coinb2: str) -> bytes:
    """Build complete coinbase transaction"""
    return hex_to_bytes(coinb1 + extranonce1 + extranonce2 + coinb2)


def compute_merkle_root(coinbase_hash: bytes, merkle_branches: List[str]) -> bytes:
    """Compute merkle root from coinbase and branches"""
    current = coinbase_hash
    for branch in merkle_branches:
        branch_bytes = hex_to_bytes(branch)
        current = sha256d(current + branch_bytes)
    return current


# =============================================================================
# BLOCK HEADER
# =============================================================================

def build_block_header(
    version: int,
    prev_block_hash: bytes,
    merkle_root: bytes,
    timestamp: int,
    nbits: int,
    nonce: int
) -> bytes:
    """Build 80-byte block header"""
    header = (
        uint32_le(version) +
        prev_block_hash +
        merkle_root +
        uint32_le(timestamp) +
        uint32_le(nbits) +
        uint32_le(nonce)
    )
    assert len(header) == 80, f"Header must be 80 bytes, got {len(header)}"
    return header


# =============================================================================
# ENCRYPTION
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
            return base64.b64encode(json.dumps(data).encode()).decode()


# =============================================================================
# MEDICAL RECORDS
# =============================================================================

@dataclass
class MedicalRecord:
    record_id: str
    patient_id: str
    timestamp: float
    record_type: str
    data: Dict
    encrypted_blob: str
    sha256_hash: str


class MerkleTree:
    def __init__(self, records: List[MedicalRecord]):
        self.leaves = [hex_to_bytes(r.sha256_hash) for r in records]
        self.root = self._compute_root()
    
    def _compute_root(self) -> bytes:
        if not self.leaves:
            return bytes(32)
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
        return bytes_to_hex(self.root)


class MedicalRecordGenerator:
    NAMES = ["Abebe Bekele", "Tigist Haile", "Dawit Gebre", "Meron Tadesse"]
    DIAGNOSES = [("Malaria", "P. falciparum"), ("Typhoid", "S. typhi"), ("TB", "Negative")]
    
    def __init__(self):
        self.patient_counter = 0
        self.record_counter = 0
    
    def generate_batch(self, count: int = 5) -> List[MedicalRecord]:
        records = []
        patient_name = self.NAMES[self.patient_counter % len(self.NAMES)]
        patient_id = f"ETH-{self.patient_counter:06d}"
        self.patient_counter += 1
        
        for i in range(count):
            self.record_counter += 1
            diagnosis, result = self.DIAGNOSES[self.record_counter % len(self.DIAGNOSES)]
            
            record_data = {
                "patient_name": patient_name,
                "patient_id": patient_id,
                "diagnosis": diagnosis,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            encrypted = CryptoEngine.encrypt(record_data, Config.MASTER_KEY)
            record_hash = hashlib.sha256(f"{self.record_counter}:{encrypted}".encode()).hexdigest()
            
            records.append(MedicalRecord(
                record_id=f"REC-{self.record_counter:08d}",
                patient_id=patient_id,
                timestamp=time.time(),
                record_type="clinical_note",
                data=record_data,
                encrypted_blob=encrypted,
                sha256_hash=record_hash
            ))
        
        return records


# =============================================================================
# VERIFIED PROOF
# =============================================================================

@dataclass
class VerifiedProof:
    proof_id: str
    batch_id: str
    job_id: str
    timestamp: float
    job_sent_at: float
    share_received_at: float
    latency_s: float
    
    # Medical
    medical_merkle_root: str
    records_sealed: int
    
    # ASIC submission
    worker_name: str
    extranonce2: str
    ntime: str
    nonce: str
    version_bits: str  # NEW: version from ASIC (version-rolling)
    
    # Coinbase
    coinb1: str
    coinb2: str
    extranonce1: str
    
    # Computed
    coinbase_hash: str
    block_merkle_root: str
    block_header_hex: str
    block_hash_hex: str
    
    # Verification
    difficulty: float
    target_hex: str
    nbits: int
    version_used: int  # NEW: actual version used in header
    hash_meets_target: bool
    verification_method: str
    verification_error: str
    estimated_hashes: int


# =============================================================================
# STRATUM BRIDGE WITH VERSION ROLLING
# =============================================================================

class StratumBridgeV6(threading.Thread):
    """Stratum bridge with VERSION ROLLING support"""
    
    def __init__(self, output_dir: Path):
        super().__init__(daemon=True)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((Config.HOST, Config.PORT))
        self.sock.listen(5)
        
        self.conn = None
        self.authorized = False
        self.running = True
        
        # Version rolling state
        self.version_rolling_enabled = False
        self.version_rolling_mask = Config.VERSION_ROLLING_MASK
        
        self.current_job = None
        self.job_lock = threading.Lock()
        
        self.proofs: List[VerifiedProof] = []
        self.proofs_lock = threading.Lock()
        
        self.share_received = threading.Event()
        self.last_submit = None
        self.last_submit_time = None
        
        self.generator = MedicalRecordGenerator()
        self.coinb1, self.coinb2 = create_coinbase_parts()
        
        self.job_counter = 0
        self.stats = {
            'jobs_sent': 0,
            'shares_received': 0,
            'shares_verified': 0,
            'shares_failed': 0
        }
    
    def run(self):
        print(f"[BRIDGE] Listening on {Config.HOST}:{Config.PORT}")
        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()
                print(f"[BRIDGE] Connection from {addr}")
                self.conn = conn
                self._handle_connection()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[BRIDGE] Error: {e}")
    
    def _handle_connection(self):
        buffer = ""
        self.conn.settimeout(0.5)
        
        try:
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
                                self._handle_message(msg)
                            except json.JSONDecodeError:
                                pass
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[BRIDGE] Recv error: {e}")
                    break
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
            self.authorized = False
    
    def _handle_message(self, msg: Dict):
        msg_id = msg.get('id')
        method = msg.get('method')
        
        if Config.DEBUG:
            print(f"[STRATUM] <- {method}: {json.dumps(msg)[:100]}...")
        
        if method == 'mining.subscribe':
            result = [
                [["mining.set_difficulty", "1"], ["mining.notify", "1"]],
                Config.EXTRANONCE1,
                Config.EXTRANONCE2_SIZE
            ]
            self._send({'id': msg_id, 'result': result, 'error': None})
            print(f"[BRIDGE] Subscribed - extranonce1={Config.EXTRANONCE1}")
        
        elif method == 'mining.configure':
            # Handle version rolling configuration
            params = msg.get('params', [])
            extensions = params[0] if params else []
            ext_params = params[1] if len(params) > 1 else {}
            
            result = {}
            if 'version-rolling' in extensions:
                self.version_rolling_enabled = True
                mask = ext_params.get('version-rolling.mask', 'ffffffff')
                self.version_rolling_mask = int(mask, 16)
                result['version-rolling'] = True
                result['version-rolling.mask'] = format(self.version_rolling_mask, '08x')
                print(f"[BRIDGE] Version rolling enabled, mask={mask}")
            
            self._send({'id': msg_id, 'result': result, 'error': None})
        
        elif method == 'mining.suggest_difficulty':
            # Acknowledge but use our own difficulty
            self._send({'id': msg_id, 'result': True, 'error': None})
        
        elif method == 'mining.authorize':
            worker = msg.get('params', ['unknown'])[0]
            self._send({'id': msg_id, 'result': True, 'error': None})
            self.authorized = True
            print(f"[BRIDGE] Authorized: {worker}")
        
        elif method == 'mining.submit':
            self._handle_submit(msg)
        
        elif method == 'mining.extranonce.subscribe':
            self._send({'id': msg_id, 'result': True, 'error': None})
    
    def _handle_submit(self, msg: Dict):
        """
        Handle mining.submit with VERSION ROLLING support.
        
        Standard params: [worker, job_id, extranonce2, ntime, nonce]
        With version rolling: [worker, job_id, extranonce2, ntime, nonce, version_bits]
        """
        params = msg.get('params', [])
        msg_id = msg.get('id')
        
        if len(params) >= 5:
            submit_data = {
                'worker_name': params[0],
                'job_id': params[1],
                'extranonce2': params[2],
                'ntime': params[3],
                'nonce': params[4],
                'version_bits': params[5] if len(params) >= 6 else None  # VERSION ROLLING!
            }
            
            self.last_submit = submit_data
            self.last_submit_time = time.time()
            self.stats['shares_received'] += 1
            
            self._send({'id': msg_id, 'result': True, 'error': None})
            self.share_received.set()
            
            version_str = f", version={params[5]}" if len(params) >= 6 else ""
            print(f"[BRIDGE] Share: job={params[1]}, nonce={params[4]}, en2={params[2]}{version_str}")
        else:
            self._send({'id': msg_id, 'result': False, 'error': [20, "Bad params", None]})
    
    def _send(self, data: Dict):
        if self.conn:
            try:
                self.conn.sendall((json.dumps(data) + '\n').encode())
            except:
                pass
    
    def set_difficulty(self, difficulty: float):
        self._send({
            'id': None,
            'method': 'mining.set_difficulty',
            'params': [difficulty]
        })
    
    def send_job(self, medical_merkle_root: str) -> Dict:
        self.job_counter += 1
        job_id = f"med{self.job_counter:06d}"
        
        ntime = int(time.time())
        ntime_hex = format(ntime, 'x')
        
        target = difficulty_to_target(Config.DIFFICULTY)
        nbits = target_to_nbits(target)
        nbits_hex = format(nbits, '08x')
        
        version_hex = format(Config.BLOCK_VERSION, '08x')
        
        prevhash_stratum = swap_endianness_32bit_words(medical_merkle_root)
        
        job = {
            'job_id': job_id,
            'prevhash_stratum': prevhash_stratum,
            'prevhash_raw': medical_merkle_root,
            'coinb1': self.coinb1,
            'coinb2': self.coinb2,
            'merkle_branches': [],
            'version': Config.BLOCK_VERSION,
            'version_hex': version_hex,
            'nbits': nbits,
            'nbits_hex': nbits_hex,
            'ntime': ntime,
            'ntime_hex': ntime_hex,
            'difficulty': Config.DIFFICULTY,
            'target': target,
            'sent_at': time.time()
        }
        
        with self.job_lock:
            self.current_job = job
        
        notify_params = [
            job_id,
            prevhash_stratum,
            self.coinb1,
            self.coinb2,
            [],
            version_hex,
            nbits_hex,
            ntime_hex,
            True
        ]
        
        self._send({'id': None, 'method': 'mining.notify', 'params': notify_params})
        self.stats['jobs_sent'] += 1
        
        print(f"[JOB] Sent {job_id}: merkle={medical_merkle_root[:16]}..., nbits={nbits_hex}")
        
        return job
    
    def verify_share(self, job: Dict, submit: Dict) -> Tuple[bool, str, Dict]:
        """
        Verify share with VERSION ROLLING support.
        """
        verification = {}
        
        try:
            # Build coinbase
            coinbase = build_coinbase(
                job['coinb1'],
                Config.EXTRANONCE1,
                submit['extranonce2'],
                job['coinb2']
            )
            coinbase_hash = sha256d(coinbase)
            verification['coinbase_hash'] = bytes_to_hex(coinbase_hash)
            
            # Merkle root - ESP-Miner reverses 32-bit words before putting in header
            block_merkle_root = compute_merkle_root(coinbase_hash, job['merkle_branches'])
            # Apply 32-bit word reversal (same as swap_endianness_32bit_words)
            merkle_for_header = hex_to_bytes(swap_endianness_32bit_words(bytes_to_hex(block_merkle_root)))
            verification['block_merkle_root'] = bytes_to_hex(block_merkle_root)
            
            # Prevhash - ESP-Miner reverses the prevhash bytes
            # The stratum format is already word-swapped, so we use it directly
            prevhash_bytes = hex_to_bytes(job['prevhash_stratum'])
            
            # Parse submission values
            ntime = int(submit['ntime'], 16)
            nonce = int(submit['nonce'], 16)
            
            # VERSION ROLLING: Use version from submit if available
            if submit.get('version_bits'):
                # The version_bits from ASIC replaces certain bits of original version
                version_bits = int(submit['version_bits'], 16)
                
                # Apply mask: keep non-rolling bits from original, use ASIC bits for rolling positions
                version_used = (job['version'] & ~self.version_rolling_mask) | (version_bits & self.version_rolling_mask)
            else:
                version_used = job['version']
            
            verification['version_used'] = version_used
            verification['version_hex'] = format(version_used, '08x')
            
            # Build header with ESP-Miner compatible format
            header = build_block_header(
                version=version_used,
                prev_block_hash=prevhash_bytes,
                merkle_root=merkle_for_header,  # Use word-reversed merkle root!
                timestamp=ntime,
                nbits=job['nbits'],
                nonce=nonce
            )
            verification['block_header_hex'] = bytes_to_hex(header)
            
            # Hash
            block_hash = sha256d(header)
            block_hash_display = bytes_to_hex(reverse_bytes(block_hash))
            verification['block_hash_hex'] = block_hash_display
            
            # Check target
            hash_int = int.from_bytes(block_hash, byteorder='little')
            verification['hash_int'] = hash_int
            
            meets_target = hash_int < job['target']
            
            if meets_target:
                return (True, "", verification)
            else:
                ratio = hash_int / job['target']
                return (False, f"Hash > target (ratio: {ratio:.2f})", verification)
        
        except Exception as e:
            return (False, f"Error: {str(e)}", verification)
    
    def seal_medical_batch(self, timeout: float = None) -> Optional[VerifiedProof]:
        if not self.authorized:
            return None
        
        timeout = timeout or Config.TIMEOUT_PER_SHARE
        
        # Generate records
        batch = self.generator.generate_batch(5)
        tree = MerkleTree(batch)
        medical_merkle_root = tree.get_root_hex()
        batch_id = f"BATCH-{int(time.time() * 1000)}"
        
        print(f"\n[SEAL] Batch {batch_id}")
        print(f"[SEAL] Medical root: {medical_merkle_root}")
        
        # Set difficulty
        self.set_difficulty(Config.DIFFICULTY)
        time.sleep(0.1)
        
        # Send job
        job = self.send_job(medical_merkle_root)
        
        # Wait for share
        self.share_received.clear()
        
        if not self.share_received.wait(timeout):
            print(f"[SEAL] Timeout")
            return None
        
        submit = self.last_submit
        submit_time = self.last_submit_time
        
        if not submit or submit['job_id'] != job['job_id']:
            print(f"[SEAL] Job mismatch")
            return None
        
        # Verify
        verified, error_msg, verification = self.verify_share(job, submit)
        
        if verified:
            method = "python_verified"
            self.stats['shares_verified'] += 1
            icon = "✓"
        else:
            method = "failed"
            self.stats['shares_failed'] += 1
            icon = "✗"
        
        latency = submit_time - job['sent_at']
        
        print(f"[VERIFY] {icon} {method.upper()}")
        print(f"[VERIFY] Latency: {latency:.2f}s")
        print(f"[VERIFY] Version used: {verification.get('version_hex', 'N/A')}")
        print(f"[VERIFY] Block hash: {verification.get('block_hash_hex', 'N/A')[:24]}...")
        if error_msg:
            print(f"[VERIFY] Error: {error_msg}")
        
        # Create proof
        proof = VerifiedProof(
            proof_id=f"proof_{int(submit_time * 1000)}",
            batch_id=batch_id,
            job_id=job['job_id'],
            timestamp=submit_time,
            job_sent_at=job['sent_at'],
            share_received_at=submit_time,
            latency_s=latency,
            medical_merkle_root=medical_merkle_root,
            records_sealed=len(batch),
            worker_name=submit['worker_name'],
            extranonce2=submit['extranonce2'],
            ntime=submit['ntime'],
            nonce=submit['nonce'],
            version_bits=submit.get('version_bits', ''),
            coinb1=job['coinb1'],
            coinb2=job['coinb2'],
            extranonce1=Config.EXTRANONCE1,
            coinbase_hash=verification.get('coinbase_hash', ''),
            block_merkle_root=verification.get('block_merkle_root', ''),
            block_header_hex=verification.get('block_header_hex', ''),
            block_hash_hex=verification.get('block_hash_hex', ''),
            difficulty=Config.DIFFICULTY,
            target_hex=format(job['target'], '064x'),
            nbits=job['nbits'],
            version_used=verification.get('version_used', job['version']),
            hash_meets_target=verified,
            verification_method=method,
            verification_error=error_msg[:500] if error_msg else "",
            estimated_hashes=int(Config.DIFFICULTY * (2**32))
        )
        
        with self.proofs_lock:
            self.proofs.append(proof)
        
        return proof
    
    def get_statistics(self) -> Dict:
        with self.proofs_lock:
            proofs = list(self.proofs)
        
        if not proofs:
            return {}
        
        verified = [p for p in proofs if p.hash_meets_target]
        latencies = [p.latency_s for p in proofs]
        
        if len(proofs) >= 2:
            total_time = proofs[-1].share_received_at - proofs[0].job_sent_at
            total_hashes = sum(p.estimated_hashes for p in proofs)
            hashrate_mhs = (total_hashes / total_time) / 1e6 if total_time > 0 else 0
        else:
            total_time = proofs[0].latency_s
            hashrate_mhs = (proofs[0].estimated_hashes / total_time) / 1e6
        
        efficiency_mhw = hashrate_mhs / Config.LV06_POWER_WATTS
        rtx3090_eff = Config.RTX3090_HASHRATE_MHS / Config.RTX3090_POWER_WATTS
        
        return {
            'experiment_date': datetime.now().isoformat(),
            'duration_s': total_time,
            'difficulty': Config.DIFFICULTY,
            'total_shares': len(proofs),
            'python_verified': len(verified),
            'verification_failed': len(proofs) - len(verified),
            'verification_rate': len(verified) / len(proofs) if proofs else 0,
            'total_records_sealed': sum(p.records_sealed for p in proofs),
            'latency_mean_s': sum(latencies) / len(latencies),
            'latency_min_s': min(latencies),
            'latency_max_s': max(latencies),
            'hashrate_mhs': hashrate_mhs,
            'efficiency_mhw': efficiency_mhw,
            'vs_rtx3090_efficiency': efficiency_mhw / rtx3090_eff if rtx3090_eff > 0 else 0,
            'power_watts': Config.LV06_POWER_WATTS,
            'version_rolling_enabled': self.version_rolling_enabled,
            'hardware': 'Lucky Miner LV06 (BM1366)',
            'encryption': 'AES-256-GCM' if HAS_CRYPTO else 'Demo'
        }
    
    def save_results(self) -> Dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.proofs_lock:
            proofs = list(self.proofs)
        
        # JSON
        proofs_file = self.output_dir / f"verified_proofs_{timestamp}.json"
        with open(proofs_file, 'w') as f:
            json.dump({
                'schema_version': '6.0',
                'version_rolling': self.version_rolling_enabled,
                'proofs': [asdict(p) for p in proofs]
            }, f, indent=2)
        print(f"[SAVE] Proofs: {proofs_file}")
        
        # CSV
        csv_file = self.output_dir / f"proofs_{timestamp}.csv"
        if proofs:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(proofs[0]).keys())
                writer.writeheader()
                for p in proofs:
                    writer.writerow(asdict(p))
        print(f"[SAVE] CSV: {csv_file}")
        
        # Stats
        stats = self.get_statistics()
        stats_file = self.output_dir / f"statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[SAVE] Stats: {stats_file}")
        
        self._print_summary(stats)
        return stats
    
    def _print_summary(self, stats: Dict):
        print("\n" + "=" * 70)
        print("                    EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"\n📊 VERIFICATION:")
        print(f"   Total shares:        {stats.get('total_shares', 0)}")
        print(f"   Python verified:     {stats.get('python_verified', 0)}")
        print(f"   Failed:              {stats.get('verification_failed', 0)}")
        print(f"   Rate:                {stats.get('verification_rate', 0) * 100:.1f}%")
        print(f"\n📁 RECORDS: {stats.get('total_records_sealed', 0)}")
        print(f"\n⏱️ LATENCY: {stats.get('latency_mean_s', 0):.2f}s mean")
        print(f"\n⚡ HASHRATE: {stats.get('hashrate_mhs', 0):.2f} MH/s")
        print(f"   Efficiency: {stats.get('efficiency_mhw', 0):.2f} MH/W")
        print(f"   vs RTX 3090: {stats.get('vs_rtx3090_efficiency', 0):.1f}×")
        print(f"\n🔄 Version rolling: {stats.get('version_rolling_enabled', False)}")
        print("=" * 70)
    
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
    print("=" * 70)
    print("  ASIC-RAG-HEALTH v6.0 - WITH VERSION ROLLING SUPPORT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Difficulty:      {Config.DIFFICULTY}")
    print(f"  Duration:        {Config.DURATION_SEC}s")
    print(f"  Version rolling: Supported (BIP320)")
    print()
    
    bridge = StratumBridgeV6(Config.OUTPUT_DIR)
    bridge.start()
    
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "localhost"
    
    print(f"Waiting for miner on {local_ip}:{Config.PORT}...")
    
    try:
        while not bridge.authorized and bridge.running:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
        return
    
    print("\n" + "=" * 70)
    print("                    EXPERIMENT STARTED")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < Config.DURATION_SEC and bridge.running:
            proof = bridge.seal_medical_batch()
            
            elapsed = time.time() - start_time
            remaining = Config.DURATION_SEC - elapsed
            
            with bridge.proofs_lock:
                n_total = len(bridge.proofs)
                n_verified = len([p for p in bridge.proofs if p.hash_meets_target])
            
            print(f"\n[PROGRESS] {elapsed:.0f}s / {Config.DURATION_SEC}s, "
                  f"{n_verified}/{n_total} verified")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    print("\n\nSaving...")
    bridge.save_results()
    bridge.stop()
    
    print(f"\nDone! Results in {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
