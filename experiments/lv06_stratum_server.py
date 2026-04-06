#!/usr/bin/env python3
"""
LV06 Stratum Server - Minimal Viable Experiment
Hardware-accelerated SHA-256 testing with Lucky Miner LV06

Purpose:
    Demonstrate that ASIC BM1366 can perform SHA-256 work for RAG systems.
    Measure REAL performance data without modifying firmware.

Protocol:
    Stratum v1 mining protocol
    Compatible with AxeOS/ESP-Miner firmware

Author: ASIC-RAG-CHIMERA Project
Date: 2024-12-18
License: MIT
"""

import socket
import json
import time
import hashlib
import threading
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import struct
from datetime import datetime


@dataclass
class MiningJob:
    """Mining job sent to LV06 ASIC"""
    job_id: str
    prev_hash: str  # 32 bytes hex (can encode custom data here)
    coinbase1: str
    coinbase2: str
    merkle_branches: List[str]
    version: str    # 4 bytes hex
    nbits: str      # 4 bytes hex (difficulty)
    ntime: str      # 4 bytes hex (timestamp)
    clean_jobs: bool

    # Metadata
    created_at: float = field(default_factory=time.time)
    data_payload: bytes = b""  # Original data we want to hash


@dataclass
class SubmittedShare:
    """Share submitted by miner"""
    job_id: str
    extranonce2: str
    ntime: str
    nonce: str
    received_at: float = field(default_factory=time.time)
    worker_name: str = ""


@dataclass
class MinerConnection:
    """Connected miner state"""
    socket: socket.socket
    address: Tuple[str, int]
    connected_at: float = field(default_factory=time.time)
    worker_name: str = ""
    shares_submitted: int = 0
    shares_accepted: int = 0
    shares_rejected: int = 0
    last_share_time: float = 0.0


class LV06StratumServer:
    """
    Minimal Stratum server for LV06 experiments.

    Features:
    - Accepts LV06 connection via Stratum protocol
    - Sends custom SHA-256 jobs (encoded in block headers)
    - Receives shares and measures throughput
    - Collects REAL performance metrics

    Usage:
        server = LV06StratumServer(host='0.0.0.0', port=3333)
        server.start()
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 3333, difficulty: int = 1):
        self.host = host
        self.port = port
        self.difficulty = difficulty

        # Server state
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.start_time: Optional[float] = None

        # Connected miners
        self.miners: Dict[Tuple[str, int], MinerConnection] = {}

        # Jobs and shares
        self.jobs: Dict[str, MiningJob] = {}
        self.shares: List[SubmittedShare] = []
        self.job_counter = 0

        # Statistics
        self.total_hashes_computed = 0
        self.total_shares_received = 0
        self.total_shares_accepted = 0
        self.total_shares_rejected = 0

        # Stratum configuration
        self.extranonce1 = "01000000"  # 4 bytes
        self.extranonce2_size = 4      # 4 bytes

        # Logging
        self.verbose = True

    def log(self, message: str, prefix: str = "INFO"):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{prefix}] {message}")

    def start(self):
        """Start the Stratum server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            self.start_time = time.time()

            print("=" * 70)
            print("LV06 STRATUM SERVER - EXPERIMENTAL")
            print("=" * 70)
            print(f"Server started: {self.host}:{self.port}")
            print(f"Difficulty: {self.difficulty}")
            print(f"ExtraNonce1: {self.extranonce1}")
            print(f"ExtraNonce2 Size: {self.extranonce2_size} bytes")
            print()
            print("Configure your LV06:")
            print(f"  Pool URL: stratum+tcp://192.168.0.14:{self.port}")
            print("  Username: test")
            print("  Password: x")
            print()
            print("Waiting for LV06 connection...")
            print("=" * 70)
            print()

            # Accept connections
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    self.log(f"Miner connected from {address[0]}:{address[1]}", "CONNECT")

                    # Create miner connection object
                    miner = MinerConnection(socket=client_socket, address=address)
                    self.miners[address] = miner

                    # Handle in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(miner,),
                        daemon=True
                    )
                    client_thread.start()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.log(f"Error accepting connection: {e}", "ERROR")

        except Exception as e:
            self.log(f"Fatal error: {e}", "FATAL")

        finally:
            self.stop()

    def handle_client(self, miner: MinerConnection):
        """Handle communication with connected miner"""
        buffer = ""

        try:
            while self.running:
                # Receive data
                data = miner.socket.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break

                buffer += data

                # Process complete messages (lines terminated by \n)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line:
                        response = self.handle_message(line, miner)
                        if response:
                            miner.socket.send((response + '\n').encode('utf-8'))

        except Exception as e:
            self.log(f"Error handling {miner.address}: {e}", "ERROR")

        finally:
            # Clean up
            try:
                miner.socket.close()
            except:
                pass

            if miner.address in self.miners:
                del self.miners[miner.address]

            self.log(f"Miner {miner.address[0]}:{miner.address[1]} disconnected", "DISCONNECT")

    def handle_message(self, message: str, miner: MinerConnection) -> Optional[str]:
        """Process Stratum message from miner"""
        try:
            msg = json.loads(message)
            method = msg.get('method')
            msg_id = msg.get('id')
            params = msg.get('params', [])

            # Log incoming message
            if method:
                self.log(f"{method} (id={msg_id})", "RX")

            # Route to handler
            if method == 'mining.subscribe':
                return self.handle_subscribe(msg_id, params)

            elif method == 'mining.authorize':
                return self.handle_authorize(msg_id, params, miner)

            elif method == 'mining.submit':
                return self.handle_submit(msg_id, params, miner)

            elif method == 'mining.extranonce.subscribe':
                return json.dumps({"id": msg_id, "result": True, "error": None})

            else:
                self.log(f"Unknown method: {method}", "WARN")
                return json.dumps({"id": msg_id, "result": None, "error": None})

        except json.JSONDecodeError:
            self.log(f"Invalid JSON: {message[:100]}", "ERROR")
            return None

        except Exception as e:
            self.log(f"Error processing message: {e}", "ERROR")
            return None

    def handle_subscribe(self, msg_id: int, params: List) -> str:
        """Handle mining.subscribe"""
        response = {
            "id": msg_id,
            "result": [
                [
                    ["mining.set_difficulty", "deadbeef"],
                    ["mining.notify", "deadbeef"]
                ],
                self.extranonce1,
                self.extranonce2_size
            ],
            "error": None
        }
        return json.dumps(response)

    def handle_authorize(self, msg_id: int, params: List, miner: MinerConnection) -> Optional[str]:
        """Handle mining.authorize and send initial job"""
        # Extract username
        username = params[0] if params else "unknown"
        miner.worker_name = username

        self.log(f"Worker authorized: {username}", "AUTH")

        # Send authorization response
        auth_response = {
            "id": msg_id,
            "result": True,
            "error": None
        }
        miner.socket.send((json.dumps(auth_response) + '\n').encode('utf-8'))

        # Send difficulty
        diff_msg = {
            "id": None,
            "method": "mining.set_difficulty",
            "params": [self.difficulty]
        }
        miner.socket.send((json.dumps(diff_msg) + '\n').encode('utf-8'))
        self.log(f"Difficulty set: {self.difficulty}", "TX")

        # Send first job
        job = self.create_job()
        self.send_job(miner, job)

        return None  # Already sent response

    def handle_submit(self, msg_id: int, params: List, miner: MinerConnection) -> str:
        """Handle mining.submit (share submission)"""
        # Parse share: [username, job_id, extranonce2, ntime, nonce]
        username = params[0] if len(params) > 0 else ""
        job_id = params[1] if len(params) > 1 else ""
        extranonce2 = params[2] if len(params) > 2 else ""
        ntime = params[3] if len(params) > 3 else ""
        nonce = params[4] if len(params) > 4 else ""

        # Create share record
        share = SubmittedShare(
            job_id=job_id,
            extranonce2=extranonce2,
            ntime=ntime,
            nonce=nonce,
            worker_name=username
        )

        self.shares.append(share)
        miner.shares_submitted += 1
        miner.last_share_time = time.time()

        # Update statistics
        self.total_shares_received += 1

        # Calculate hashes (each share at difficulty=1 represents 2^32 hashes)
        hashes = self.difficulty * (2 ** 32)
        self.total_hashes_computed += hashes

        # Accept share (in real pool, would validate)
        self.total_shares_accepted += 1
        miner.shares_accepted += 1

        # Calculate current hashrate
        elapsed = time.time() - self.start_time
        current_hashrate = self.total_hashes_computed / elapsed if elapsed > 0 else 0

        # Log share
        share_num = len(self.shares)
        self.log(
            f"Share #{share_num} | Job: {job_id} | Nonce: {nonce} | "
            f"Hashrate: {current_hashrate/1e9:.2f} GH/s | "
            f"Total: {self.total_hashes_computed/1e9:.2f} GH",
            "SHARE"
        )

        # Send new job every 10 shares to keep miner busy
        if share_num % 10 == 0:
            job = self.create_job()
            self.send_job(miner, job)

        # Return acceptance
        response = {
            "id": msg_id,
            "result": True,
            "error": None
        }

        return json.dumps(response)

    def create_job(self, custom_data: Optional[bytes] = None) -> MiningJob:
        """
        Create mining job with optional custom data.

        The custom data (up to 32 bytes) is encoded in the prev_hash field.
        This allows us to test ASIC performance with controlled data.
        """
        self.job_counter += 1
        job_id = f"job_{self.job_counter:08d}"

        # Generate or use custom data
        if custom_data is None:
            # Generate test data: hash of job_id
            custom_data = hashlib.sha256(job_id.encode()).digest()

        # Encode custom data in prev_hash (32 bytes hex = 64 chars)
        prev_hash = custom_data.hex().ljust(64, '0')[:64]

        # Coinbase transaction (simplified)
        coinbase1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
        coinbase2 = "ffffffff0100f2052a01000000434104"

        # Block header fields
        version = "20000000"
        nbits = "1d00ffff"  # Very low difficulty for fast shares
        ntime = hex(int(time.time()))[2:].zfill(8)

        job = MiningJob(
            job_id=job_id,
            prev_hash=prev_hash,
            coinbase1=coinbase1,
            coinbase2=coinbase2,
            merkle_branches=[],
            version=version,
            nbits=nbits,
            ntime=ntime,
            clean_jobs=True,
            data_payload=custom_data
        )

        self.jobs[job_id] = job
        return job

    def send_job(self, miner: MinerConnection, job: MiningJob):
        """Send mining job to miner"""
        notify_msg = {
            "id": None,
            "method": "mining.notify",
            "params": [
                job.job_id,
                job.prev_hash,
                job.coinbase1,
                job.coinbase2,
                job.merkle_branches,
                job.version,
                job.nbits,
                job.ntime,
                job.clean_jobs
            ]
        }

        miner.socket.send((json.dumps(notify_msg) + '\n').encode('utf-8'))
        self.log(f"Job sent: {job.job_id} | Data: {job.data_payload.hex()[:32]}...", "TX")

    def get_statistics(self) -> Dict:
        """Get current server statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        hashrate = self.total_hashes_computed / elapsed if elapsed > 0 else 0

        return {
            "uptime_seconds": elapsed,
            "total_shares": self.total_shares_received,
            "accepted_shares": self.total_shares_accepted,
            "rejected_shares": self.total_shares_rejected,
            "total_hashes": self.total_hashes_computed,
            "hashrate_hs": hashrate,
            "hashrate_ghs": hashrate / 1e9,
            "connected_miners": len(self.miners),
            "jobs_created": self.job_counter
        }

    def print_statistics(self):
        """Print current statistics"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"Uptime:           {stats['uptime_seconds']:.2f} seconds")
        print(f"Shares received:  {stats['total_shares']}")
        print(f"Shares accepted:  {stats['accepted_shares']}")
        print(f"Jobs created:     {stats['jobs_created']}")
        print(f"Total hashes:     {stats['total_hashes']/1e9:.2f} GH")
        print(f"Hashrate:         {stats['hashrate_ghs']:.2f} GH/s")
        print(f"Connected miners: {stats['connected_miners']}")
        print("=" * 70)

    def stop(self):
        """Stop the server"""
        self.log("Shutting down server...", "SHUTDOWN")
        self.running = False

        # Close all miner connections
        for miner in list(self.miners.values()):
            try:
                miner.socket.close()
            except:
                pass

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        # Print final statistics
        self.print_statistics()


def main():
    """Run the LV06 Stratum server"""
    print("\n" + "=" * 70)
    print("LV06 STRATUM SERVER - HARDWARE VALIDATION EXPERIMENT")
    print("=" * 70)
    print()
    print("Purpose:")
    print("  Demonstrate that ASIC BM1366 can perform SHA-256 work for RAG.")
    print("  Measure REAL performance without firmware modification.")
    print()
    print("Setup Instructions:")
    print("  1. Access LV06 web interface: http://192.168.0.15")
    print("  2. Go to: Settings → Pool Configuration")
    print("  3. Set Pool 1:")
    print("     - URL: stratum+tcp://192.168.0.14:3333")
    print("     - Username: test")
    print("     - Password: x")
    print("  4. Save and wait for connection")
    print()
    print("Press Ctrl+C to stop server and view statistics")
    print("=" * 70)
    print()

    # Create and start server
    server = LV06StratumServer(host='0.0.0.0', port=3333, difficulty=1)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n")
        server.stop()


if __name__ == "__main__":
    main()
