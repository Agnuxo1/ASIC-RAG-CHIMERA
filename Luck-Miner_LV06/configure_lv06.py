#!/usr/bin/env python3
"""
LV06 Auto-Configuration Script
Automatically configures Lucky Miner LV06 for CHIMERA experiments

Usage:
    python configure_lv06.py --miner-ip 192.168.0.15 --pc-ip 192.168.0.14
    python configure_lv06.py --miner-ip 192.168.0.15 --auto  # Auto-detect PC IP

Example:
    python configure_lv06.py --miner-ip 192.168.0.15 --pc-ip 192.168.0.14 --voltage 990 --frequency 400
"""

import argparse
import json
import socket
import urllib.request
import urllib.error
import time
import sys

def get_local_ip():
    """Auto-detect PC's local IP address"""
    try:
        # Create a socket to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def http_request(url, method="GET", data=None, timeout=5):
    """Make HTTP request to LV06"""
    try:
        if data:
            data = json.dumps(data).encode('utf-8')

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Content-Type', 'application/json')

        response = urllib.request.urlopen(req, timeout=timeout)
        return True, response.read().decode()
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except Exception as e:
        return False, str(e)

def check_connectivity(miner_ip):
    """Verify miner is reachable"""
    print(f"🔍 Checking connectivity to {miner_ip}...")

    url = f"http://{miner_ip}/api/system/info"
    success, response = http_request(url)

    if success:
        data = json.loads(response)
        print(f"   ✅ Miner online")
        print(f"      Voltage:   {data.get('voltage', data.get('coreVoltageActual', 'N/A'))} mV")
        print(f"      Frequency: {data.get('frequency', 'N/A')} MHz")
        print(f"      Temp:      {data.get('temp', 'N/A')}°C")
        print(f"      Hash Rate: {data.get('hashRate', 0):.2f} GH/s")
        return True, data
    else:
        print(f"   ❌ Cannot connect: {response}")
        print(f"      → Verify miner IP is correct")
        print(f"      → Check miner is powered on")
        print(f"      → Try: ping {miner_ip}")
        return False, None

def configure_pool(miner_ip, pc_ip, port=3333):
    """Configure pool to point to CHIMERA bridge"""
    print(f"\n⚙️  Configuring pool...")

    pool_url = f"stratum+tcp://{pc_ip}:{port}"
    print(f"   Target: {pool_url}")

    # Note: Pool config endpoint varies by firmware
    # Try multiple methods

    # Method 1: Direct pool endpoint (some firmware versions)
    url = f"http://{miner_ip}/api/pool"
    payload = {
        "url": pool_url,
        "user": "chimera",
        "pass": "x"
    }

    success, response = http_request(url, method="PATCH", data=payload)

    if success:
        print(f"   ✅ Pool configured via /api/pool")
        return True

    # Method 2: System config (other firmware versions)
    url = f"http://{miner_ip}/api/system"
    payload = {
        "stratumURL": pool_url,
        "stratumUser": "chimera",
        "stratumPassword": "x"
    }

    success, response = http_request(url, method="PATCH", data=payload)

    if success:
        print(f"   ✅ Pool configured via /api/system")
        return True

    # If both fail, instruct manual config
    print(f"   ⚠️  Auto-config failed: {response}")
    print(f"\n   📋 Manual Configuration Required:")
    print(f"      1. Open browser: http://{miner_ip}")
    print(f"      2. Navigate to Pool Settings")
    print(f"      3. Set Pool 1:")
    print(f"         URL:      {pool_url}")
    print(f"         User:     chimera")
    print(f"         Password: x")
    print(f"      4. Click Save")

    return False

def configure_voltage_frequency(miner_ip, voltage, frequency):
    """Set voltage and frequency"""
    print(f"\n⚙️  Configuring hardware parameters...")
    print(f"   Voltage:   {voltage} mV")
    print(f"   Frequency: {frequency} MHz")

    url = f"http://{miner_ip}/api/system"
    payload = {
        "frequency": int(frequency),
        "volts": int(voltage)
    }

    success, response = http_request(url, method="PATCH", data=payload)

    if success:
        print(f"   ✅ Hardware parameters updated")
        print(f"      NOTE: Requires miner restart to take effect")
        return True
    else:
        print(f"   ❌ Configuration failed: {response}")
        return False

def restart_miner(miner_ip):
    """Restart miner to apply settings"""
    print(f"\n🔄 Restarting miner...")

    url = f"http://{miner_ip}/api/system/restart"

    success, response = http_request(url, method="POST", data={}, timeout=3)

    if success or "timed out" in str(response).lower():
        # Timeout is expected (miner reboots immediately)
        print(f"   ✅ Restart command sent")
        print(f"   ⏳ Waiting 40 seconds for reboot...")

        for i in range(40, 0, -5):
            print(f"      {i}s remaining...", end="\r")
            time.sleep(5)

        print("\n   ✅ Reboot complete")
        return True
    else:
        print(f"   ❌ Restart failed: {response}")
        print(f"      → Manually restart via web UI or power cycle")
        return False

def verify_configuration(miner_ip, expected_voltage, expected_frequency):
    """Verify settings were applied"""
    print(f"\n✅ Verifying configuration...")

    url = f"http://{miner_ip}/api/system/info"
    success, response = http_request(url)

    if not success:
        print(f"   ⚠️  Cannot verify (miner not responding)")
        return False

    data = json.loads(response)
    actual_voltage = data.get('voltage', data.get('coreVoltageActual', 0))
    actual_frequency = data.get('frequency', 0)

    print(f"   Voltage:   {actual_voltage} mV (expected: {expected_voltage} mV)")
    print(f"   Frequency: {actual_frequency} MHz (expected: {expected_frequency} MHz)")

    voltage_match = abs(actual_voltage - expected_voltage) < 50  # ±50 mV tolerance
    frequency_match = abs(actual_frequency - expected_frequency) < 50  # ±50 MHz tolerance

    if voltage_match and frequency_match:
        print(f"   ✅ Configuration verified successfully")
        return True
    else:
        print(f"   ⚠️  Configuration mismatch")
        if not voltage_match:
            print(f"      Voltage off by {abs(actual_voltage - expected_voltage)} mV")
        if not frequency_match:
            print(f"      Frequency off by {abs(actual_frequency - expected_frequency)} MHz")
        return False

def main():
    parser = argparse.ArgumentParser(description='Configure LV06 for CHIMERA experiments')
    parser.add_argument('--miner-ip', required=True, help='LV06 IP address (e.g., 192.168.0.15)')
    parser.add_argument('--pc-ip', help='PC IP address (e.g., 192.168.0.14)')
    parser.add_argument('--auto', action='store_true', help='Auto-detect PC IP')
    parser.add_argument('--port', type=int, default=3333, help='Stratum port (default: 3333)')
    parser.add_argument('--voltage', type=int, default=990, help='Voltage in mV (default: 990)')
    parser.add_argument('--frequency', type=int, default=400, help='Frequency in MHz (default: 400)')
    parser.add_argument('--no-restart', action='store_true', help='Skip automatic restart')

    args = parser.parse_args()

    # Determine PC IP
    if args.auto:
        pc_ip = get_local_ip()
        if not pc_ip:
            print("❌ Auto-detection failed. Please specify --pc-ip manually")
            sys.exit(1)
        print(f"🔍 Auto-detected PC IP: {pc_ip}")
    elif args.pc_ip:
        pc_ip = args.pc_ip
    else:
        print("❌ Error: Specify --pc-ip or use --auto")
        sys.exit(1)

    print("=" * 60)
    print("LV06 AUTO-CONFIGURATION FOR CHIMERA")
    print("=" * 60)
    print(f"Miner IP:   {args.miner_ip}")
    print(f"PC IP:      {pc_ip}")
    print(f"Port:       {args.port}")
    print(f"Voltage:    {args.voltage} mV")
    print(f"Frequency:  {args.frequency} MHz")
    print()

    # Step 1: Check connectivity
    success, current_config = check_connectivity(args.miner_ip)
    if not success:
        sys.exit(1)

    # Step 2: Configure pool
    if not configure_pool(args.miner_ip, pc_ip, args.port):
        print("\n⚠️  Pool configuration may need manual setup")

    # Step 3: Configure voltage/frequency
    if not configure_voltage_frequency(args.miner_ip, args.voltage, args.frequency):
        sys.exit(1)

    # Step 4: Restart (if not skipped)
    if not args.no_restart:
        if not restart_miner(args.miner_ip):
            print("\n⚠️  Please restart miner manually")
            sys.exit(1)

        # Step 5: Verify
        verify_configuration(args.miner_ip, args.voltage, args.frequency)
    else:
        print("\n⚠️  Skipped restart (--no-restart). Changes require manual reboot.")

    # Final instructions
    print("\n" + "=" * 60)
    print("CONFIGURATION COMPLETE")
    print("=" * 60)
    print("\n✅ Next Steps:")
    print(f"   1. Start CHIMERA bridge:")
    print(f"      cd \"D:\\Holographic Reservoir Computing\\V04\"")
    print(f"      python drivers/chronos_bridge.py")
    print()
    print(f"   2. Wait for connection:")
    print(f"      You should see: ⚡ ASIC CONNECTED: {args.miner_ip}")
    print()
    print(f"   3. Verify metrics:")
    print(f"      Shares should appear as dots: ..........")
    print()
    print(f"   4. Run experiments:")
    print(f"      python experiments/exp_01_voltage_modulation.py")
    print()
    print(f"📊 Monitor miner status: http://{args.miner_ip}")

if __name__ == "__main__":
    main()
