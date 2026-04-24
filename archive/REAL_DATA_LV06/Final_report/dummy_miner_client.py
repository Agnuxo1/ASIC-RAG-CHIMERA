#!/usr/bin/env python3
"""Simple dummy miner client to connect to the LV06StratumServer for testing.
It mimics the minimal Stratum protocol required by the experiment script.
"""
import socket
import json
import time
import threading

HOST = "127.0.0.1"  # server runs on localhost
PORT = 3333

def recv_thread(sock):
    buffer = ""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                try:
                    msg = json.loads(line)
                    # Print received messages for debugging
                    print("[SERVER]", msg)
                except Exception:
                    pass
        except Exception:
            break

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    threading.Thread(target=recv_thread, args=(s,), daemon=True).start()
    # Subscribe
    s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": []}) + "\n").encode())
    time.sleep(0.5)
    # Authorize
    s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": ["user", "pass"]}) + "\n").encode())
    time.sleep(0.5)
    share_id = 3
    while True:
        # Send a dummy share submission
        submit_msg = {
            "id": share_id,
            "method": "mining.submit",
            "params": ["worker", "jobid", "extranonce2", "ntime", "nonce"]
        }
        s.sendall((json.dumps(submit_msg) + "\n").encode())
        share_id += 1
        time.sleep(0.1)  # 10 Hz share rate

if __name__ == "__main__":
    main()
