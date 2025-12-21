import socket
import json
import time
import threading
import health_config as cfg

class MedicalBridge:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", cfg.STRATUM_PORT))
        self.sock.listen(5)
        
        # Telemetry Buffer
        self.telemetry = {
            "start_time": time.time(),
            "shares_accepted": 0,
            "current_difficulty": 1,
            "last_share_time": 0,
            "hashrate_window": []
        }
        self.lock = threading.Lock()
        
        print(f"🏥 ASIC-RAG-HEALTH BRIDGE ONLINE")
        print(f"   Mode: Physical Hardware Link (LV06)")
        print(f"   Listening for Clinic Node at: {cfg.PC_IP}:{cfg.STRATUM_PORT}")

        # Start API Thread
        threading.Thread(target=self.api_server, daemon=True).start()
        self.accept_miners()

    def api_server(self):
        api = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        api.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        api.bind(("0.0.0.0", cfg.API_PORT))
        api.listen(5)
        
        while True:
            try:
                conn, _ = api.accept()
                cmd = conn.recv(1024).decode().strip()
                
                if cmd == "GET_STATS":
                    with self.lock:
                        # Calculate instantaneous hashrate based on shares
                        payload = json.dumps(self.telemetry)
                    conn.sendall(payload.encode())
                
                elif cmd == "RESET":
                    with self.lock:
                        self.telemetry["shares_accepted"] = 0
                        self.telemetry["start_time"] = time.time()
                        self.telemetry["hashrate_window"] = []
                    conn.sendall(b"OK")
                
                conn.close()
            except Exception as e:
                print(f"API Error: {e}")

    def accept_miners(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"⚡ CLINIC NODE CONNECTED: {addr[0]} (LV06)")
            threading.Thread(target=self.handle_miner, args=(conn,)).start()

    def handle_miner(self, conn):
        buffer = ""
        try:
            while True:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    msg_str, buffer = buffer.split('\n', 1)
                    if not msg_str.strip(): continue
                    self.process_stratum(json.loads(msg_str), conn)
        except:
            print("🔌 Clinic Node Disconnected")

    def process_stratum(self, msg, conn):
        msg_id = msg.get('id')
        method = msg.get('method')
        
        if method == 'mining.subscribe':
            resp = {"id": msg_id, "result": [[["mining.set_difficulty", "1"], ["mining.notify", "1"]], "08000002", 4], "error": None}
            self.send(conn, resp)
            
        elif method == 'mining.authorize':
            resp = {"id": msg_id, "result": True, "error": None}
            self.send(conn, resp)
            self.send(conn, {"id": None, "method": "mining.set_difficulty", "params": [cfg.BLOCK_DIFFICULTY]})
            self.send_job(conn)
            
        elif method == 'mining.submit':
            with self.lock:
                self.telemetry["shares_accepted"] += 1
                self.telemetry["last_share_time"] = time.time()
            
            # Heartbeat for the user
            print("✚", end="", flush=True) 
            self.send(conn, {"id": msg_id, "result": True, "error": None})

    def send_job(self, conn):
        # We send a static job. In a real deployment, this would be the Merkle Root of patient records.
        job_id = "medical_block_001"
        msg = {
            "params": [job_id, "0"*64, "01"*32, "0000", [], "20000000", "1d00ffff", hex(int(time.time()))[2:], True],
            "id": None, "method": "mining.notify"
        }
        self.send(conn, msg)

    def send(self, conn, data):
        try:
            conn.sendall((json.dumps(data) + '\n').encode())
        except: pass

if __name__ == "__main__":
    MedicalBridge()