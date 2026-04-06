#!/usr/bin/env python3
"""
Working Stratum Server for LV06
Based on proven chimera_chronos.py implementation

This version has been tested and WORKS with LV06 hardware.
"""

import socket
import json
import time
import threading
import hashlib

# Configuration
HOST_IP = "0.0.0.0"
PORT = 3333
DIFFICULTY = 1

class WorkingStratumServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST_IP, PORT))
        self.sock.listen(5)
        self.shares_received = 0
        self.total_hashes = 0
        self.start_time = time.time()
        print(f"=== WORKING STRATUM SERVER ===")
        print(f"Listening on {HOST_IP}:{PORT}")
        print(f"Difficulty: {DIFFICULTY}")
        print(f"Waiting for LV06 connection...")
        print()

    def handle_client(self, conn, addr):
        print(f"[CONNECTED] {addr[0]}:{addr[1]}")
        buffer = ""

        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    break

                buffer += data.decode('utf-8', errors='ignore')

                while '\n' in buffer:
                    msg_str, buffer = buffer.split('\n', 1)
                    if not msg_str.strip():
                        continue

                    try:
                        msg = json.loads(msg_str)
                        self.process_message(msg, conn)
                    except Exception as e:
                        print(f"[ERROR] Failed to process message: {e}")

            except Exception as e:
                print(f"[ERROR] Connection error: {e}")
                break

        print(f"[DISCONNECTED] {addr[0]}:{addr[1]}")

    def process_message(self, msg, conn):
        method = msg.get('method')
        msg_id = msg.get('id')

        if method == 'mining.subscribe':
            print(f"[RX] mining.subscribe (id={msg_id})")
            # IMPORTANT: This exact format works with AxeOS
            resp = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "1"], ["mining.notify", "1"]],
                    "08000002",  # extranonce1
                    4            # extranonce2_size
                ],
                "error": None
            }
            self.send_json(conn, resp)
            print(f"[TX] Subscribe response sent")

        elif method == 'mining.authorize':
            username = msg.get('params', ['unknown'])[0]
            print(f"[RX] mining.authorize: {username}")

            # Send authorization response
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)
            print(f"[TX] Authorization accepted")

            # Set difficulty
            self.set_difficulty(conn, DIFFICULTY)

            # Send first job
            self.send_job(conn)

        elif method == 'mining.submit':
            self.shares_received += 1
            self.total_hashes += (2 ** 32)  # Each share at diff=1 is 2^32 hashes

            elapsed = time.time() - self.start_time
            hashrate_ghs = (self.total_hashes / elapsed) / 1e9 if elapsed > 0 else 0

            print(f"[SHARE #{self.shares_received}] Hashrate: {hashrate_ghs:.2f} GH/s | Total: {self.total_hashes/1e9:.2f} GH")

            # Accept share
            resp = {"id": msg_id, "result": True, "error": None}
            self.send_json(conn, resp)

            # Send new job every 5 shares
            if self.shares_received % 5 == 0:
                self.send_job(conn)

    def set_difficulty(self, conn, diff):
        msg = {
            "id": None,
            "method": "mining.set_difficulty",
            "params": [diff]
        }
        self.send_json(conn, msg)
        print(f"[TX] Difficulty set to {diff}")

    def send_job(self, conn):
        job_id = f"job_{int(time.time())}"

        # Generate custom data (this is where we can inject RAG data)
        custom_data = hashlib.sha256(job_id.encode()).digest()
        prev_hash = custom_data.hex()[:64].ljust(64, '0')

        # IMPORTANT: This exact parameter format works with AxeOS
        msg = {
            "params": [
                job_id,              # job_id
                prev_hash,           # prevhash (can encode custom data here!)
                "a" * 64,            # coinb1
                "0" * 64,            # coinb2
                [],                  # merkle_branch
                "20000000",          # version
                "1d00ffff",          # nbits (difficulty)
                hex(int(time.time()))[2:].zfill(8),  # ntime
                True                 # clean_jobs
            ],
            "id": None,
            "method": "mining.notify"
        }
        self.send_json(conn, msg)
        print(f"[TX] Job sent: {job_id} | Custom data: {custom_data.hex()[:32]}...")

    def send_json(self, conn, data):
        conn.sendall((json.dumps(data) + '\n').encode())

    def start(self):
        try:
            while True:
                conn, addr = self.sock.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("\n\n=== SERVER STATISTICS ===")
            elapsed = time.time() - self.start_time
            print(f"Runtime: {elapsed:.1f} seconds")
            print(f"Shares received: {self.shares_received}")
            print(f"Total hashes: {self.total_hashes/1e9:.2f} GH")
            if elapsed > 0:
                print(f"Average hashrate: {(self.total_hashes/elapsed)/1e9:.2f} GH/s")
            print("=" * 30)
        finally:
            self.sock.close()

if __name__ == "__main__":
    server = WorkingStratumServer()
    server.start()
