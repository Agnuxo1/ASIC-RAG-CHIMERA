#!/usr/bin/env python3
"""
Simple LV06 Connection Test - Windows Compatible
Tests basic connectivity without emoji characters
"""

import sys
import socket
import json
import urllib.request
import urllib.error

def test_network(ip):
    """Test basic network connectivity"""
    print(f"[1/4] Testing network connectivity to {ip}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((ip, 80))
        sock.close()

        if result == 0:
            print("   [OK] Network connection successful (port 80 open)")
            return True
        else:
            print(f"   [FAIL] Cannot connect to port 80 (error code: {result})")
            print("      - Check if LV06 is powered on")
            print(f"      - Verify IP address: {ip}")
            return False
    except Exception as e:
        print(f"   [FAIL] Connection failed: {e}")
        return False

def test_http_api(ip):
    """Test HTTP API endpoint"""
    print(f"\n[2/4] Testing HTTP API...")
    try:
        url = f"http://{ip}/api/system/info"
        response = urllib.request.urlopen(url, timeout=5)
        data = json.loads(response.read().decode())

        print("   [OK] HTTP API responding")
        print(f"      Temperature:  {data.get('temp', 'N/A')} C")
        print(f"      Voltage:      {data.get('voltage', data.get('coreVoltageActual', 'N/A'))} mV")
        print(f"      Frequency:    {data.get('frequency', 'N/A')} MHz")
        print(f"      Hash Rate:    {data.get('hashRate', 0):.2f} GH/s")
        print(f"      Power:        {data.get('power', 'N/A')} W")
        return True, data
    except urllib.error.URLError as e:
        print(f"   [FAIL] HTTP API not responding: {e}")
        print(f"      - Try accessing: http://{ip}")
        return False, None
    except json.JSONDecodeError:
        print("   [FAIL] Invalid JSON response")
        return False, None

def test_hashing(data):
    """Check if miner is actively hashing"""
    print(f"\n[3/4] Checking hashing status...")

    if not data:
        print("   [WARN] No system data available")
        return True

    hashrate = data.get('hashRate', 0)
    temp = data.get('temp', 0)

    if hashrate > 0:
        print(f"   [OK] Miner is hashing: {hashrate:.2f} GH/s")
        if temp < 50:
            print(f"      [WARN] Temperature is low ({temp} C) - might not be at full speed")
        elif temp > 80:
            print(f"      [WARN] Temperature is high ({temp} C) - check cooling!")
        else:
            print(f"      Temperature OK: {temp} C")
        return True
    else:
        print(f"   [FAIL] Miner is NOT hashing (hashrate = 0)")
        print("      - Check pool connection")
        print("      - Verify pool URL")
        print(f"      Temperature: {temp} C (should be >50C when hashing)")
        return False

def test_bridge_compatibility(ip):
    """Test if CHIMERA bridge is accessible"""
    print(f"\n[4/4] Testing CHIMERA bridge compatibility...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 4029))
        sock.close()

        if result == 0:
            print("   [OK] CHIMERA bridge is running (port 4029)")
            return True
        else:
            print("   [WARN] CHIMERA bridge not detected (port 4029 closed)")
            print("      - Bridge might not be running")
            print("      - This is OK if you haven't started it yet")
            return True  # Don't fail - bridge might not be needed right now

    except Exception as e:
        print(f"   [INFO] Bridge check: {e}")
        return True

def main():
    # Get IP from command line or use default
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.0.15"
        print(f"No IP provided, using default: {ip}")
        print(f"Usage: python {sys.argv[0]} <LV06_IP>\n")

    print("=" * 60)
    print("LV06 CONNECTION TEST - SIMPLE VERSION")
    print("=" * 60)
    print(f"Target: {ip}")
    print()

    # Run tests
    results = []

    # Test 1: Network
    results.append(test_network(ip))
    if not results[-1]:
        print("\n[CRITICAL] Cannot reach miner. Fix network connection first.")
        sys.exit(1)

    # Test 2: HTTP API
    api_ok, system_data = test_http_api(ip)
    results.append(api_ok)
    if not api_ok:
        print("\n[CRITICAL] API not responding. Check firmware.")
        sys.exit(1)

    # Test 3: Hashing
    results.append(test_hashing(system_data))

    # Test 4: Bridge
    results.append(test_bridge_compatibility(ip))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed - LV06 is ready!")
        print("\nNext steps:")
        print(f"  1. Configure pool to point to: <YOUR_PC_IP>:3333")
        print(f"  2. Start bridge: python V04/drivers/chronos_bridge.py")
        print(f"  3. Run experiments")
    elif passed >= 2:
        print("\n[OK] Basic connectivity works - Check warnings above")
    else:
        print("\n[FAIL] Tests failed - See errors above")

    sys.exit(0 if passed >= 2 else 1)

if __name__ == "__main__":
    main()
