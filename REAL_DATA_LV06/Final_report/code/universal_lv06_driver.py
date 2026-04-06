"""
Lucky Miner LV06 Universal Driver SDK v1.0
Hardware Support: Bitmain BM1387 (AxeOS Firmware) or BM1366
Purpose: High-performance Neuromorphic Reservoir Computing Control
"""

import socket
import json
import time
import struct
import requests
import threading
from typing import Optional, List, Dict, Any, Callable

class LV06Config:
    """Manages HTTP API interactions for device configuration."""
    def __init__(self, ip: str):
        self.ip = ip
        self.base_url = f"http://{ip}/api"

    def set_frequency(self, mhz: int) -> bool:
        """Sets the ASIC frequency and reboots if necessary."""
        try:
            r = requests.patch(f"{self.base_url}/system", json={"frequency": mhz}, timeout=5)
            return r.status_code == 200
        except Exception as e:
            print(f"[LV06] Config Error: {e}")
            return False

    def reboot(self):
        """Triggers a hardware reboot."""
        try:
            requests.post(f"{self.base_url}/reboot", timeout=2)
            return True
        except:
            return False

class LV06StratumServer(threading.Thread):
    """
    High-fidelity Stratum Server for state injection and harvesting.
    Optimized for low-latency neuromorphic feedback.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 3333):
        super().__init__()
        self.host = host
        self.port = port
        self.running = True
        self.connection_active = False
        self.client_conn = None
        self.current_shares = []
        self.job_counter = 0
        
        # Stratum Constants
        self.extranonce1 = "08000002"
        self.extranonce2_size = 4
        self.handshake_complete = threading.Event()

    def run(self):
        """Main server loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(1.0)
        
        print(f"[Stratum] Listening on {self.host}:{self.port}")
        
        while self.running:
            try:
                conn, addr = sock.accept()
                print(f"[STRATUM] Accepted connection from {addr}", flush=True)
                self._handle_client(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running: print(f"[STRATUM] Socket Error: {e}", flush=True)

    def _handle_client(self, conn):
        self.client_conn = conn
        self.connection_active = True
        print(f"[Stratum] Client Handshake Start", flush=True)
        buffer = ""
        while self.running:
            try:
                data_bytes = conn.recv(8192)
                if not data_bytes: 
                    print("[Stratum] Client Disconnected", flush=True)
                    break
                print(f"[RAW RECV] {data_bytes.hex()}", flush=True)
                data = data_bytes.decode('utf-8', errors='ignore')
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_line(conn, line)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Stratum] Read Error: {e}", flush=True)
                break
        self.connection_active = False
        self.client_conn = None

    def _process_line(self, conn, line):
        print(f"[RECV] {line.strip()}", flush=True)
        try:
            msg = json.loads(line)
        except: return
        
        mid = msg.get('id')
        method = msg.get('method')
        
        if method == 'mining.subscribe':
            res = [[["mining.set_difficulty", "s1"], ["mining.notify", "s2"]], self.extranonce1, self.extranonce2_size]
            self._send(conn, {"id": mid, "result": res, "error": None})
        elif method == 'mining.authorize':
            # Send initial difficulty immediately to wake up the chip
            self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [512.0]})
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.handshake_complete.set()
        elif method == 'mining.submit':
            # Capture share with high-precision timestamp
            arrival = time.perf_counter()
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.current_shares.append({"time": arrival, "msg": msg})
            print(f"[SHARE] Captured at {arrival:.4f}", flush=True)
        elif method == 'mining.configure':
            self._send(conn, {"id": mid, "result": {"version-rolling.mask": "ffffffff"}, "error": None})

    def _send(self, conn, data):
        try:
            line = json.dumps(data) + "\n"
            print(f"[SEND] {line.strip()}", flush=True)
            conn.sendall(line.encode())
        except Exception as e:
            print(f"[SEND ERROR] {e}", flush=True)

    def set_difficulty(self, diff: float):
        """Updates the difficulty target for the miner."""
        if self.client_conn:
            self._send(self.client_conn, {"id": None, "method": "mining.set_difficulty", "params": [diff]})

    def inject_rate(self, value: float, d_base: float = 1.0, epsilon: float = 0.01):
        """
        Rate-Encoded Reservoir Computing: Modulates the share arrival rate
        by dynamically changing the mining difficulty based on input u[t].
        """
        if not self.client_conn: return False
        
        # WAIT for handshake before first injection
        if not self.handshake_complete.is_set():
            print("[RE-RC] Waiting for handshake...")
            if not self.handshake_complete.wait(timeout=10):
                print("[RE-RC] Handshake TIMEOUT")
                return False
        
        target_diff = max(0.0001, d_base / (value + epsilon))
        self.set_difficulty(target_diff)
        print(f"[RE-RC] Injected: {value:.3f} -> Target Difficulty: {target_diff:.4f}", flush=True)
        
        self.job_counter += 1
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        # RESTORING TRUSTED COINBASE FORMAT
        header = "01000000" + "01" + "00" * 32 + "ffffffff" + "10" 
        coinb1 = header + "04" + "00"*8 + "0a" + "00"*10 # Dummy 0s in input slot
        coinb2 = "ffffffff" + "01" + "00f2052a01000000" + "00"*8
        
        # USE clean_jobs=True for immediate Rate-Encoded modulation
        params = [str(self.job_counter), "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        self._send(self.client_conn, {"id": None, "method": "mining.notify", "params": params})
        return True

    def harvest_state(self):
        """Returns and clears all shares captured since last call."""
        batch = list(self.current_shares)
        self.current_shares = []
        return batch

    def stop(self):
        self.running = False
        if self.client_conn: self.client_conn.close()

if __name__ == "__main__":
    # Self-test / Example usage
    print("LV06 Driver SDK initialized.")
