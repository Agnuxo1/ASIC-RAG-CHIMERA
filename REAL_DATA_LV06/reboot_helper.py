import requests
import time

IP = "192.168.0.15"

def reboot():
    endpoints = [
        f"http://{IP}/api/reboot",
        f"http://{IP}/api/restart", 
        f"http://{IP}/api/admin/reboot"
    ]
    
    for ep in endpoints:
        print(f"Trying {ep}...")
        try:
            r = requests.post(ep, timeout=2)
            print(f"Response: {r.status_code}")
            if r.status_code == 200:
                print("Reboot command accepted.")
                return
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    reboot()
