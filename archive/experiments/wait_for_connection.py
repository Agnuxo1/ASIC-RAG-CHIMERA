#!/usr/bin/env python3
"""Wait for LV06 to connect and start mining"""

import time
import urllib.request
import json

def get_stats():
    try:
        response = urllib.request.urlopen('http://192.168.0.15/api/system/info', timeout=5)
        return json.loads(response.read().decode())
    except Exception as e:
        return None

print("Waiting for LV06 to reconnect and start mining...")
print("="*70)

for i in range(12):  # 2 minutes total
    stats = get_stats()
    if stats:
        hashrate = stats.get('hashRate', 0)
        shares = stats.get('sharesAccepted', 0)
        wifi = stats.get('wifiStatus', '')
        pool = stats.get('stratumURL', '')
        port = stats.get('stratumPort', 0)

        print(f"[{i+1:2d}/12] Hashrate: {hashrate:6.2f} GH/s | Shares: {shares:3d} | WiFi: {wifi:20s} | Pool: {pool}:{port}")

        if hashrate > 0:
            print("="*70)
            print("SUCCESS! LV06 is mining!")
            print("="*70)
            break
    else:
        print(f"[{i+1:2d}/12] Cannot connect to LV06 (still restarting...)")

    time.sleep(10)
else:
    print("="*70)
    print("LV06 did not start mining after 2 minutes")
    print("="*70)
