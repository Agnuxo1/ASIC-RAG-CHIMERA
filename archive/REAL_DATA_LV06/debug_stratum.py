import socket
import json
import time

HOST = "0.0.0.0"
PORT = 3333

def raw_listener():
    print(f"Listening on {HOST}:{PORT}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    
    conn, addr = s.accept()
    print(f"Connected by {addr}")
    
    while True:
        data = conn.recv(4096)
        if not data:
            break
        print(f"RAW RECV: {data}")
        # Send minimal responses to keep it alive
        try:
            for line in data.decode().split('\n'):
                if not line.strip(): continue
                msg = json.loads(line)
                mid = msg.get('id')
                method = msg.get('method')
                if method == 'mining.subscribe':
                    resp = {"id": mid, "result": [[["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]], "08000002", 4], "error": None}
                    conn.sendall((json.dumps(resp) + '\n').encode())
                    # Send easy difficulty
                    diff = {"id": None, "method": "mining.set_difficulty", "params": [0.1]}
                    conn.sendall((json.dumps(diff) + '\n').encode())
                elif method == 'mining.authorize':
                    resp = {"id": mid, "result": True, "error": None}
                    conn.sendall((json.dumps(resp) + '\n').encode())
                    # Send a job
                    job = {"id": None, "method": "mining.notify", "params": ["1", "0"*64, "0100000001"+"00"*32+"ffffffff10"+"00"*16, "ffffffff0100f2052a01000000"+"00"*8, [], "20000000", "1f00ffff", hex(int(time.time()))[2:], True]}
                    conn.sendall((json.dumps(job) + '\n').encode())
                elif method == 'mining.submit':
                    resp = {"id": mid, "result": True, "error": None}
                    conn.sendall((json.dumps(resp) + '\n').encode())
                    print("!!! SHARE RECEIVED !!!")
        except Exception as e:
            print(f"Parse error: {e}")

if __name__ == "__main__":
    raw_listener()
