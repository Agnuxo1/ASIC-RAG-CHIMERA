import socket
import time

HOST = "0.0.0.0"
PORT = 3333

def debug_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"DEBUG: Listening on {PORT}...")
    
    conn, addr = s.accept()
    print(f"DEBUG: Accepted connection from {addr}")
    conn.settimeout(10)
    
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("DEBUG: Connection closed by peer")
                break
            print(f"DEBUG RECV: {data}")
            # Try to send a simple mining.set_difficulty to see if it responds
            # But normally we wait for subscribe
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
    finally:
        conn.close()
        s.close()

if __name__ == "__main__":
    debug_socket()
