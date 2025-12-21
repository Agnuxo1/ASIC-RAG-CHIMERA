# chronos_bridge_v2.py
import socket
import json
import time
import threading
import binascii
import struct
import lab_config as cfg

class ChronosBridge:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", cfg.STRATUM_PORT))
        self.sock.listen(5)
        
        # Buffer de Datos Reales (Time Anchor)
        self.share_buffer = [] 
        self.buffer_lock = threading.Lock()
        self.total_shares = 0
        self.start_time = time.time()
        
        print(f"⏳ CHRONOS BRIDGE V2 (REAL HARDWARE LINK) ONLINE")
        print(f"   Escuchando Stratum en: {cfg.PC_IP}:{cfg.STRATUM_PORT}")
        print(f"   API de Datos en:       0.0.0.0:{cfg.BRIDGE_API_PORT}")

        # Iniciar hilos
        threading.Thread(target=self.api_server, daemon=True).start()
        self.accept_miners()

    def api_server(self):
        api = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        api.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        api.bind(("0.0.0.0", cfg.BRIDGE_API_PORT))
        api.listen(5)
        
        while True:
            try:
                conn, _ = api.accept()
                req = conn.recv(1024).decode().strip()
                
                if req == "GET_DATA":
                    # BATCH FETCH: Entregar datos y limpiar buffer (Atomic)
                    with self.buffer_lock:
                        payload = json.dumps({
                            "uptime": time.time() - self.start_time,
                            "total_shares": self.total_shares,
                            "timestamps": self.share_buffer
                        })
                        self.share_buffer = [] # Limpiar tras entrega
                    
                    conn.sendall(payload.encode())
                
                elif req == "RESET":
                    with self.buffer_lock:
                        self.share_buffer = []
                        self.total_shares = 0
                        self.start_time = time.time()
                    conn.sendall(b"OK")
                    
                conn.close()
            except Exception as e:
                print(f"API Error: {e}")

    def accept_miners(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"⚡ ASIC CONECTADO: {addr[0]}")
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
            print("🔌 Minero desconectado")

    def process_stratum(self, msg, conn):
        msg_id = msg.get('id')
        method = msg.get('method')
        
        # Respuesta estándar Stratum
        if method == 'mining.subscribe':
            resp = {"id": msg_id, "result": [[["mining.set_difficulty", "1"], ["mining.notify", "1"]], "08000002", 4], "error": None}
            self.send(conn, resp)
            
        elif method == 'mining.authorize':
            resp = {"id": msg_id, "result": True, "error": None}
            self.send(conn, resp)
            # Dificultad baja para maximizar flujo de datos (Entropía)
            self.send(conn, {"id": None, "method": "mining.set_difficulty", "params": [1]})
            self.send_job(conn)
            
        elif method == 'mining.submit':
            # CAPTURA DE DATO REAL
            arrival_time = time.time()
            with self.buffer_lock:
                self.share_buffer.append(arrival_time)
                self.total_shares += 1
                if len(self.share_buffer) > 20000: # Safety cap
                    self.share_buffer.pop(0)
            
            print(".", end="", flush=True) # Latido visual
            self.send(conn, {"id": msg_id, "result": True, "error": None})

    def send_job(self, conn):
        # Job estático para mantener el flujo
        job_id = "feed_me_entropy"
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
    ChronosBridge()