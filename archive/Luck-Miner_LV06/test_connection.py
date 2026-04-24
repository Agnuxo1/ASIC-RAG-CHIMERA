#!/usr/bin/env python3
"""
LV06 Connection Test Script
Tests connectivity and basic functionality of Lucky Miner LV06

Usage:
    python test_connection.py [LV06_IP]

Example:
    python test_connection.py 192.168.0.15
"""

import sys
import socket
import json
import urllib.request
import urllib.error

def test_ping(ip):
    """Test basic network connectivity"""
    print(f"[1/5] Testing network connectivity to {ip}...")
    try:
        # Try to connect to HTTP port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((ip, 80))
        sock.close()

        if result == 0:
            print("   ✅ Network connection OK (port 80 open)")
            return True
        else:
            print(f"   ❌ Cannot connect to port 80 (error code: {result})")
            print("      → Check if LV06 is powered on")
            print("      → Verify IP address is correct")
            print(f"      → Try pinging: ping {ip}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def test_http_api(ip):
    """Test HTTP API endpoint"""
    print(f"\n[2/5] Testing HTTP API...")
    try:
        url = f"http://{ip}/api/system/info"
        response = urllib.request.urlopen(url, timeout=5)
        data = json.loads(response.read().decode())

        print("   ✅ HTTP API responding")
        print(f"      Temperature: {data.get('temp', 'N/A')}°C")
        print(f"      Voltage:     {data.get('voltage', data.get('coreVoltageActual', 'N/A'))} mV")
        print(f"      Frequency:   {data.get('frequency', 'N/A')} MHz")
        print(f"      Hash Rate:   {data.get('hashRate', 0):.2f} GH/s")
        print(f"      Power:       {data.get('power', 'N/A')} W")
        return True, data
    except urllib.error.URLError as e:
        print(f"   ❌ HTTP API not responding: {e}")
        print("      → Check if AxeOS firmware is running")
        print(f"      → Try accessing web UI: http://{ip}")
        return False, None
    except json.JSONDecodeError:
        print("   ❌ Invalid JSON response (firmware issue?)")
        return False, None

def test_pool_config(ip):
    """Check pool configuration"""
    print(f"\n[3/5] Checking pool configuration...")
    try:
        # Note: Pool endpoint varies by firmware version
        # Try common endpoints
        endpoints = ["/api/pool", "/api/system/info"]
        pool_info = None

        for endpoint in endpoints:
            try:
                url = f"http://{ip}{endpoint}"
                response = urllib.request.urlopen(url, timeout=3)
                data = json.loads(response.read().decode())
                if 'poolUrl' in data or 'pool' in data:
                    pool_info = data
                    break
            except:
                continue

        if pool_info:
            pool_url = pool_info.get('poolUrl', pool_info.get('pool', {}).get('url', 'Unknown'))
            print(f"   ✅ Pool configured: {pool_url}")
            return True
        else:
            print("   ⚠️  Could not read pool config (check web UI)")
            print(f"      → Manually verify at: http://{ip}")
            return True  # Don't fail, just warn

    except Exception as e:
        print(f"   ⚠️  Pool config check failed: {e}")
        return True

def test_hashing(data):
    """Check if miner is actively hashing"""
    print(f"\n[4/5] Checking hashing status...")

    if not data:
        print("   ⚠️  No system data available")
        return True

    hashrate = data.get('hashRate', 0)
    temp = data.get('temp', 0)

    if hashrate > 0:
        print(f"   ✅ Miner is hashing: {hashrate:.2f} GH/s")
        if temp < 50:
            print(f"      ⚠️  Temperature is low ({temp}°C) - might not be running at full speed")
        elif temp > 80:
            print(f"      ⚠️  Temperature is high ({temp}°C) - check cooling!")
        else:
            print(f"      Temperature OK: {temp}°C")
        return True
    else:
        print(f"   ❌ Miner is NOT hashing (hashrate = 0)")
        print("      → Check pool connection")
        print("      → Verify pool URL is correct")
        print(f"      → Try restarting miner")
        print(f"      Temperature: {temp}°C (should be >50°C when hashing)")
        return False

def test_chimera_bridge_compatibility(ip):
    """Test if settings are compatible with CHIMERA bridge"""
    print(f"\n[5/5] Testing CHIMERA bridge compatibility...")

    try:
        # Try to connect to bridge API (if running)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 4029))
        sock.close()

        if result == 0:
            print("   ✅ CHIMERA bridge is running (port 4029)")

            # Try to get metrics
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect(('127.0.0.1', 4029))
                s.sendall(b'GET_METRICS')
                data = s.recv(4096).decode()
                s.close()

                metrics = json.loads(data)
                cv = metrics.get('cv', 0)
                timestamp = metrics.get('timestamp', 0)

                if timestamp > 0 and cv > 0:
                    print(f"      Bridge receiving data: CV={cv:.4f}")
                    print("      ✅ FULL INTEGRATION CONFIRMED")
                elif timestamp > 0:
                    print(f"      Bridge active but no CV yet (timestamp={timestamp})")
                    print("      → Wait for 10+ shares to accumulate")
                else:
                    print("      Bridge responding but no data yet")
                    print("      → Verify miner pool points to this PC")
            except:
                print("   ⚠️  Bridge running but no metrics yet")
                print("      → This is normal if just started")
        else:
            print("   ⚠️  CHIMERA bridge not detected (port 4029 closed)")
            print("      → Start bridge: python V04/drivers/chronos_bridge.py")
            print("      → Or bridge is running on different port")

        return True
    except Exception as e:
        print(f"   ⚠️  Bridge check failed: {e}")
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
    print("LV06 CONNECTION TEST")
    print("=" * 60)
    print(f"Target: {ip}")
    print()

    # Run tests
    results = []

    # Test 1: Ping
    results.append(test_ping(ip))
    if not results[-1]:
        print("\n❌ CRITICAL: Cannot reach miner. Fix network connection first.")
        sys.exit(1)

    # Test 2: HTTP API
    api_ok, system_data = test_http_api(ip)
    results.append(api_ok)
    if not api_ok:
        print("\n❌ CRITICAL: API not responding. Check firmware.")
        sys.exit(1)

    # Test 3: Pool config
    results.append(test_pool_config(ip))

    # Test 4: Hashing
    results.append(test_hashing(system_data))

    # Test 5: Bridge compatibility
    results.append(test_chimera_bridge_compatibility(ip))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - LV06 is ready for CHIMERA experiments!")
        print("\nNext steps:")
        print("  1. Ensure pool points to: <YOUR_PC_IP>:3333")
        print("  2. Start bridge: python V04/drivers/chronos_bridge.py")
        print("  3. Run experiment: python V04/experiments/exp_01_voltage_modulation.py")
    elif passed >= 3:
        print("\n⚠️  MOSTLY OK - Some warnings, but should work")
        print("   Check warnings above and fix if needed")
    else:
        print("\n❌ TESTS FAILED - Fix issues before running experiments")
        print("   See error messages above for solutions")

    sys.exit(0 if passed >= 3 else 1)

if __name__ == "__main__":
    main()
