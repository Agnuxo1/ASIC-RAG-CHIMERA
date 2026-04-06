import requests
import json
import time

IP = "192.168.0.15"
TARGET = 525

def get_current_freq():
    endpoints = ["/api/system", "/api/config", "/api/stats"]
    for ep in endpoints:
        try:
            r = requests.get(f"http://{IP}{ep}", headers={"Accept": "application/json"}, timeout=2)
            if r.status_code == 200:
                data = r.json()
                # Bitaxe/AxeOS usually has this in 'frequency'
                if 'frequency' in data:
                    return int(data['frequency'])
                elif 'freq' in data:
                    return int(data['freq'])
        except:
            pass
    return None

def set_freq(freq):
    print(f"Attempting to set frequency to {freq} MHz...")
    try:
        # AxeOS uses PATCH
        r = requests.patch(f"http://{IP}/api/system", json={"frequency": freq}, timeout=5)
        print(f"  PATCH /api/system: {r.status_code}")
        # Reboot to apply
        requests.post(f"http://{IP}/api/reboot", timeout=2)
        print("  Reboot command sent.")
        return True
    except Exception as e:
        print(f"  Error setting frequency: {e}")
        return False

def monitor_mhz():
    print("--- MHz MONITORING & ESCALONAMIENTO ---")
    current = get_current_freq()
    print(f"Initial Frequency: {current if current else 'Unknown'}")
    
    if current != TARGET:
        # Step-by-step or direct? The user said "escalonar si no se realizan"
        # I'll try direct first, then if it doesn't change, I'll step.
        set_freq(TARGET)
        print("Waiting 45s for reboot and stability...")
        time.sleep(45)
        
        new_freq = get_current_freq()
        print(f"New Frequency: {new_freq if new_freq else 'Unknown'}")
        
        if new_freq != TARGET:
            print("Direct change failed. Starting escalonamiento...")
            # Simple scaling: start from 400 and go up by 25
            for f in range(400, TARGET + 1, 25):
                print(f"Stepping to {f} MHz...")
                set_freq(f)
                time.sleep(40)
    else:
        print(f"Already at target frequency {TARGET} MHz.")

if __name__ == "__main__":
    monitor_mhz()
