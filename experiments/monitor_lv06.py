#!/usr/bin/env python3
"""Monitor LV06 performance in real-time"""

import time
import urllib.request
import json

def get_stats():
    try:
        response = urllib.request.urlopen('http://192.168.0.15/api/system/info', timeout=5)
        return json.loads(response.read().decode())
    except Exception as e:
        return None

print("Monitoring LV06 for 2 minutes (12 samples, 10s interval)...")
print("="*70)

samples = []
for i in range(12):
    stats = get_stats()
    if stats:
        hashrate = stats.get('hashRate', 0)
        shares = stats.get('sharesAccepted', 0)
        temp = stats.get('temp', 0)
        power = stats.get('power', 0)

        samples.append({
            'hashrate': hashrate,
            'shares': shares,
            'temp': temp,
            'power': power
        })

        print(f"[{i+1:2d}/12] Hashrate: {hashrate:6.2f} GH/s | Shares: {shares:3d} | Temp: {temp:2d}C | Power: {power:5.2f}W")
    else:
        print(f"[{i+1:2d}/12] ERROR - Cannot connect to LV06")

    if i < 11:  # Don't sleep on last iteration
        time.sleep(10)

# Calculate averages
if samples:
    avg_hashrate = sum(s['hashrate'] for s in samples) / len(samples)
    avg_power = sum(s['power'] for s in samples) / len(samples)
    max_shares = max(s['shares'] for s in samples)

    print("="*70)
    print("SUMMARY:")
    print(f"  Average Hashrate: {avg_hashrate:.2f} GH/s")
    print(f"  Average Power:    {avg_power:.2f}W")
    print(f"  Total Shares:     {max_shares}")
    print(f"  Efficiency:       {avg_hashrate/avg_power:.2f} GH/W")
    print("="*70)
