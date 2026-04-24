#!/usr/bin/env python3
"""Simple Stratum test server - ASCII only for Windows compatibility"""

import socket
import json
import time
import hashlib
import threading

class SimpleStratumServer:
    def __init__(self):
        self.server_socket = None
        self.running = False
        self.shares_received = 0
        self.total_hashes = 0
        self.start_time = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', 3333))
        self.server_socket.listen(5)
        self.running = True
        self.start_time = time.time()

        print("="*70)
        print("STRATUM SERVER RUNNING ON PORT 3333")
        print("="*70)
        print("Waiting for LV06 to connect...")
        print()

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                print(f"CONNECTED: {address[0]}:{address[1]}")

                thread = threading.Thread(target=self.handle_client, args=(client_socket, address))
                thread.daemon = True
                thread.start()
            except KeyboardInterrupt:
                break

    def handle_client(self, client_socket, address):
        buffer = ""

        try:
            while self.running:
                data = client_socket.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break

                buffer += data

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line:
                        response = self.handle_message(line, client_socket)
                        if response:
                            client_socket.send((response + '\n').encode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client_socket.close()
            print(f"DISCONNECTED: {address[0]}:{address[1]}")

    def handle_message(self, message, client_socket):
        try:
            msg = json.loads(message)
            method = msg.get('method')
            msg_id = msg.get('id')
            params = msg.get('params', [])

            print(f"RX: {method}")

            if method == 'mining.subscribe':
                response = {
                    "id": msg_id,
                    "result": [
                        [["mining.set_difficulty", "deadbeef"], ["mining.notify", "deadbeef"]],
                        "01000000",
                        4
                    ],
                    "error": None
                }
                return json.dumps(response)

            elif method == 'mining.authorize':
                username = params[0] if params else "unknown"
                print(f"AUTHORIZED: {username}")

                auth_response = {
                    "id": msg_id,
                    "result": True,
                    "error": None
                }
                client_socket.send((json.dumps(auth_response) + '\n').encode('utf-8'))

                # Send difficulty
                diff_msg = {
                    "id": None,
                    "method": "mining.set_difficulty",
                    "params": [1]
                }
                client_socket.send((json.dumps(diff_msg) + '\n').encode('utf-8'))

                # Send job
                job_id = f"job_{int(time.time())}"
                custom_data = hashlib.sha256(job_id.encode()).digest()

                notify_msg = {
                    "id": None,
                    "method": "mining.notify",
                    "params": [
                        job_id,
                        custom_data.hex().ljust(64, '0')[:64],  # prev_hash
                        "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff",
                        "ffffffff0100f2052a01000000434104",
                        [],
                        "20000000",
                        "1d00ffff",
                        hex(int(time.time()))[2:].zfill(8),
                        True
                    ]
                }
                client_socket.send((json.dumps(notify_msg) + '\n').encode('utf-8'))
                print(f"JOB SENT: {job_id}")

                return None

            elif method == 'mining.submit':
                self.shares_received += 1
                self.total_hashes += (2 ** 32)

                elapsed = time.time() - self.start_time
                hashrate = self.total_hashes / elapsed if elapsed > 0 else 0

                print(f"SHARE #{self.shares_received} | Hashrate: {hashrate/1e9:.2f} GH/s")

                return json.dumps({"id": msg_id, "result": True, "error": None})

            else:
                return json.dumps({"id": msg_id, "result": None, "error": None})

        except Exception as e:
            print(f"Error processing message: {e}")
            return None

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Shares received: {self.shares_received}")
        print(f"Total hashes: {self.total_hashes/1e9:.2f} GH")
        if elapsed > 0:
            print(f"Hashrate: {self.total_hashes/elapsed/1e9:.2f} GH/s")
        print("="*70)

if __name__ == "__main__":
    server = SimpleStratumServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()
