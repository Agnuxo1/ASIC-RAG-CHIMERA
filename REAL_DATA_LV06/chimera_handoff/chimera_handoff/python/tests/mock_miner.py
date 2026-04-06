#!/usr/bin/env python3
import socket
import json
import time

def mock_miner():
    print("🤖 STARTING MOCK MINER...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 3333))
        
        # 1. Subscribe
        s.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
        print(" [SENT] subscribe")
        print(" [RECV]", s.recv(1024).decode())
        
        # 2. Authorize
        s.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": ["user", "pass"]}).encode() + b"\n")
        print(" [SENT] authorize")
        print(" [RECV]", s.recv(1024).decode())
        
        # 3. Wait for job and submit "share"
        while True:
            data = s.recv(4096).decode()
            if not data: break
            for line in data.split("\n"):
                if not line.strip(): continue
                msg = json.loads(line)
                if msg.get("method") == "mining.notify":
                    job_id = msg["params"][0]
                    print(f" [JOB] Received: {job_id}")
                    # Simulate finding a share after a short delay
                    time.sleep(1.0)
                    submit = {
                        "id": 10,
                        "method": "mining.submit",
                        "params": ["user", job_id, "00000000", "00000000", "nonce123"]
                    }
                    s.sendall(json.dumps(submit).encode() + b"\n")
                    print(f" [SENT] submit for {job_id}")
                elif msg.get("id") == 10:
                    print(f" [RECV] share result: {msg.get('result')}")
                    
    except Exception as e:
        print(f"❌ Mock miner error: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    mock_miner()
