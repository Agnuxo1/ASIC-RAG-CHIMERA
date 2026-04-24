import requests
import json
import time

IP = "192.168.0.15"

def probe():
    endpoints = [
        "/api/system",
        "/api/status",
        "/cgi-bin/get_miner_status.cgi",
        "/cgi-bin/minerStatus.cgi",
        "/stats"
    ]
    
    for ep in endpoints:
        url = f"http://{IP}{ep}"
        try:
            print(f"Probing {url}...")
            r = requests.get(url, timeout=2)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                print(f"  Result: {r.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    probe()
