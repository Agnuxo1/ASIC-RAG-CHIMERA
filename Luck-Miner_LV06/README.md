# Lucky Miner LV06 - Complete Setup Guide

**Hardware**: Lucky Miner LV06 (BM1387 ASIC Chip - Same as Antminer S9)
**Firmware**: AxeOS
**Purpose**: Thermodynamic Entropy Source for CHIMERA Reservoir Computing System
**Last Updated**: 2025-12-18

---

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Initial Setup](#initial-setup)
3. [Network Configuration](#network-configuration)
4. [Pool Configuration](#pool-configuration)
5. [AxeOS HTTP API Reference](#axeos-http-api-reference)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## Hardware Overview

### Specifications

| Component | Specification |
|-----------|--------------|
| **ASIC Chip** | BM1387 (Same as Antminer S9) |
| **Hash Rate** | ~500-600 GH/s @ 400MHz |
| **Power Consumption** | 40-50W typical |
| **Voltage Range** | 850-1200 mV (Safe: 900-1100 mV) |
| **Frequency Range** | 50-600 MHz (Recommended: 300-500 MHz) |
| **Temperature Range** | 0-85°C (Optimal: 50-70°C) |
| **Network** | WiFi 802.11 b/g/n |
| **Firmware** | AxeOS (ESP-Miner based) |

### Physical Connections

```
[Power Supply 5V/12V] ──> [DC Jack]
                              │
                         [LV06 Board]
                              │
                        [BM1387 ASIC]
                              │
                         [Heat Sink]
                              │
                          [Cooling Fan]
```

**IMPORTANT**: Ensure adequate cooling. The BM1387 can reach 80°C+ under full load.

---

## Initial Setup

### Step 1: Power On

1. Connect **5V/12V power supply** (check board specs for exact voltage)
2. Wait 10-15 seconds for boot
3. Look for **blue/red LED** indicating WiFi AP mode (if first boot)

### Step 2: Connect to WiFi AP

**If LV06 is in AP mode (default on first boot):**

1. On your PC/Phone, search for WiFi network: `LuckyMiner-XXXX` or `AxeOS-XXXX`
2. Connect with password: `password` or `12345678` (check your specific firmware)
3. Open browser: `http://192.168.4.1`

### Step 3: Configure WiFi Client Mode

1. In AxeOS web interface: **Settings → WiFi**
2. Select your home/office WiFi network
3. Enter WiFi password
4. Click **Save & Reboot**
5. Wait 30 seconds for miner to reconnect to your network

### Step 4: Find Miner IP

**Option A - Router Admin Panel:**
- Login to your router (usually `192.168.0.1` or `192.168.1.1`)
- Look for device named `LuckyMiner` or `AxeOS`
- Note the IP address (example: `192.168.0.15`)

**Option B - Network Scanner:**
```bash
# Windows
arp -a | findstr "192.168"

# Linux/Mac
sudo nmap -sn 192.168.0.0/24
# Or
arp-scan --localnet
```

**Option C - Serial Console (Advanced):**
- Connect USB-to-Serial adapter to debug pins
- Use PuTTY/minicom at 115200 baud
- IP address will be printed on boot

---

## Network Configuration

### Assign Static IP (Recommended)

**Why**: Prevents IP changes that break CHIMERA bridge connection

**Method 1 - Router DHCP Reservation:**
1. Find LV06's MAC address in router
2. Reserve IP (example: `192.168.0.15`)
3. Reboot miner

**Method 2 - AxeOS Web Interface:**
1. Access `http://<miner_ip>`
2. **Settings → Network → Static IP**
3. Set:
   - IP Address: `192.168.0.15`
   - Subnet Mask: `255.255.255.0`
   - Gateway: `192.168.0.1` (your router IP)
   - DNS: `8.8.8.8`
4. Save & Reboot

### Verify Connection

```bash
# Ping test
ping 192.168.0.15

# HTTP API test
curl http://192.168.0.15/api/system/info
```

**Expected Response:**
```json
{
  "temp": 65,
  "voltage": 990,
  "frequency": 400,
  "power": 45,
  "hashRate": 500.5
}
```

---

## Pool Configuration

### For CHIMERA Experiments (Local Stratum Server)

**Scenario**: You're running `chronos_bridge.py` or `plenum_bridge.py` on your PC

1. Access AxeOS web interface: `http://192.168.0.15`
2. Navigate to: **Pool Settings**
3. Configure Pool 1:
   ```
   URL:      stratum+tcp://192.168.0.14:3333
   User:     chimera
   Password: x
   ```
   (Replace `192.168.0.14` with your PC's IP)

4. **Disable** Pool 2 and Pool 3 (or leave as backup)
5. Click **Save**
6. Miner will disconnect from current pool and reconnect to your bridge

### For Standard Bitcoin Mining (Optional)

**Pool 1 Example (SlushPool):**
```
URL:      stratum+tcp://stratum.slushpool.com:3333
User:     your_username.worker_name
Password: x
```

**Pool 2 Example (Backup):**
```
URL:      stratum+tcp://backup.pool.com:3333
User:     your_username
Password: x
```

---

## AxeOS HTTP API Reference

### Base URL
```
http://192.168.0.15
```

### Endpoints

#### 1. Get System Info
```bash
GET /api/system/info
```

**Response:**
```json
{
  "temp": 65,
  "voltage": 990,
  "frequency": 400,
  "power": 45.2,
  "hashRate": 523.4,
  "bestDiff": 128,
  "freeHeap": 123456,
  "coreVoltageActual": 990,
  "uptimeSeconds": 3600
}
```

#### 2. Update System Settings (Voltage/Frequency)
```bash
PATCH /api/system
Content-Type: application/json

{
  "frequency": 400,
  "volts": 990
}
```

**Example (curl):**
```bash
curl -X PATCH http://192.168.0.15/api/system \
  -H "Content-Type: application/json" \
  -d '{"frequency": 400, "volts": 990}'
```

**Example (Python):**
```python
import requests
import json

url = "http://192.168.0.15/api/system"
payload = {"frequency": 400, "volts": 990}

response = requests.patch(url, json=payload)
print(response.status_code)  # 200 = Success
```

**IMPORTANT**: After PATCH, you must **restart** the miner for changes to take effect.

#### 3. Restart System
```bash
POST /api/system/restart
Content-Type: application/json

{}
```

**Example:**
```bash
curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Note**: Miner will reboot (takes ~30-40 seconds) and reconnect to pool.

#### 4. Get WiFi Info
```bash
GET /api/system/wifi
```

**Response:**
```json
{
  "ssid": "YourWiFiName",
  "rssi": -45,
  "ip": "192.168.0.15"
}
```

---

## Troubleshooting

### Problem 1: Cannot Access Web Interface

**Symptoms:**
- Browser shows "Connection Refused" or timeout
- Ping fails

**Solutions:**

1. **Check Power**:
   - LED should be lit (blue/red)
   - Measure voltage at DC jack (5V or 12V depending on model)

2. **Verify IP**:
   - Use router admin panel to find current IP
   - Try default AP mode: `http://192.168.4.1`

3. **Reset to AP Mode**:
   - Hold reset button for 10 seconds
   - Connect to `LuckyMiner-XXXX` WiFi
   - Reconfigure from `192.168.4.1`

4. **Firewall**:
   - Disable Windows Firewall temporarily
   - Check router firewall rules

### Problem 2: Miner Not Hashing (0 GH/s)

**Symptoms:**
- Hash rate shows 0 or very low
- Temperature stays at ambient (~25-30°C)

**Solutions:**

1. **Check Pool Connection**:
   ```bash
   curl http://192.168.0.15/api/system/info
   ```
   - If `hashRate: 0`, pool is not sending jobs

2. **Verify Pool Config**:
   - Ensure URL is correct (including port :3333)
   - For CHIMERA: Verify `chronos_bridge.py` is running
   - Check bridge logs: Should show `⚡ ASIC CONNECTED`

3. **Frequency Too Low**:
   - If frequency < 100 MHz, increase to 300-400 MHz
   - Use API: `PATCH /api/system` with `{"frequency": 400}`

4. **Voltage Too Low**:
   - If voltage < 900 mV, ASIC may not start
   - Increase to 990 mV (default)

### Problem 3: High Temperature (>80°C)

**Symptoms:**
- Temp reading >80°C
- Thermal throttling (hash rate drops)

**Solutions:**

1. **Check Fan**:
   - Fan should spin at boot
   - Replace if failed

2. **Reduce Frequency**:
   - Lower from 400 MHz → 350 MHz
   - Temperature should drop 5-10°C

3. **Reduce Voltage**:
   - Lower from 990 mV → 950 mV
   - Monitor hash rate (should drop slightly)

4. **Improve Airflow**:
   - Remove obstructions around heat sink
   - Add external fan

### Problem 4: Miner Keeps Rebooting

**Symptoms:**
- LED blinks repeatedly
- Uptime resets every 30-60 seconds

**Solutions:**

1. **Power Supply**:
   - Check voltage stability (use multimeter)
   - Insufficient current → upgrade PSU (2A+ recommended)

2. **Overheating**:
   - If temp >85°C at reboot, ASIC auto-protects
   - Improve cooling

3. **Corrupt Firmware**:
   - Reflash AxeOS via serial/USB
   - Download latest from: `https://github.com/skot/ESP-Miner`

### Problem 5: WiFi Disconnects Frequently

**Symptoms:**
- Miner loses connection every few minutes
- Pool shows miner offline intermittently

**Solutions:**

1. **Signal Strength**:
   - Check RSSI: `GET /api/system/wifi`
   - If RSSI < -70 dBm, move closer to router or add WiFi extender

2. **Router Settings**:
   - Increase DHCP lease time (24 hours+)
   - Disable "Smart Connect" (forces 2.4GHz)
   - Disable "Client Isolation"

3. **Static IP**:
   - Assign static IP to prevent DHCP renewal issues

---

## Advanced Configuration

### Voltage/Frequency Optimization

**Goal**: Maximize efficiency (GH/W) or hash rate

| Voltage (mV) | Frequency (MHz) | Hash Rate (GH/s) | Power (W) | Efficiency (GH/W) |
|--------------|-----------------|------------------|-----------|-------------------|
| 850 | 300 | ~350 | 30 | 11.7 |
| 900 | 350 | ~450 | 38 | 11.8 |
| 950 | 400 | ~520 | 45 | 11.6 |
| 990 (default) | 400 | ~550 | 48 | 11.5 |
| 1100 | 500 | ~650 | 60 | 10.8 |
| 1200 | 600 | ~700 | 75 | 9.3 |

**Recommendation for CHIMERA**:
- **Default**: 990 mV @ 400 MHz (balanced)
- **Efficiency**: 900 mV @ 350 MHz (cool & quiet)
- **Max Hash**: 1100 mV @ 500 MHz (hot & loud)

### Custom Firmware Compilation

**Prerequisites**:
- ESP-IDF v4.4+
- Python 3.8+
- Git

**Steps**:
```bash
# Clone AxeOS
git clone https://github.com/skot/ESP-Miner.git
cd ESP-Miner

# Install dependencies
pip install -r requirements.txt

# Configure
idf.py menuconfig
# Navigate to: Component Config → ESP-Miner → Set defaults

# Build
idf.py build

# Flash (via serial)
idf.py -p COM3 flash
# Or upload .bin via web interface: Settings → Firmware Update
```

### Serial Console Access

**Hardware**:
- USB-to-UART adapter (3.3V logic!)
- Jumper wires

**Connections**:
```
LV06 Board          USB-UART
----------          --------
GND        <------>  GND
TX         <------>  RX
RX         <------>  TX
```

**Software**:
```bash
# Windows (PuTTY)
Port: COM3
Baud: 115200
Data: 8N1

# Linux/Mac (minicom)
minicom -D /dev/ttyUSB0 -b 115200
```

**Useful Commands**:
```
> help               # List commands
> system info        # Show system info
> wifi status        # WiFi connection status
> asic freq 400      # Set frequency
> asic voltage 990   # Set voltage
> restart            # Reboot
```

### Monitoring with Prometheus/Grafana (Optional)

**Exporter Script** (`lv06_exporter.py`):
```python
from prometheus_client import start_http_server, Gauge
import requests
import time

# Metrics
temp = Gauge('lv06_temperature_celsius', 'ASIC Temperature')
hashrate = Gauge('lv06_hashrate_ghs', 'Hash Rate')
power = Gauge('lv06_power_watts', 'Power Consumption')

def collect_metrics():
    r = requests.get('http://192.168.0.15/api/system/info')
    data = r.json()
    temp.set(data['temp'])
    hashrate.set(data['hashRate'])
    power.set(data['power'])

if __name__ == '__main__':
    start_http_server(9100)
    while True:
        collect_metrics()
        time.sleep(5)
```

**Run**:
```bash
python lv06_exporter.py
# Metrics available at: http://localhost:9100
```

---

## Safety Warnings

⚠️ **DO NOT**:
- Exceed 1200 mV voltage (risk of permanent damage)
- Run above 85°C for extended periods
- Block airflow to heat sink
- Use power supply <2A current capacity
- Touch ASIC/heat sink while powered (burn hazard)

✅ **DO**:
- Monitor temperature regularly
- Start with default settings (990 mV, 400 MHz)
- Test voltage changes incrementally (±50 mV steps)
- Keep firmware updated
- Maintain adequate cooling

---

## Specifications Summary

```
┌─────────────────────────────────────────┐
│  Lucky Miner LV06 - Quick Reference     │
├─────────────────────────────────────────┤
│ Default IP:        192.168.0.15         │
│ Web Interface:     http://192.168.0.15  │
│ Default Pool Port: 3333 (Stratum)       │
│ Default Voltage:   990 mV               │
│ Default Frequency: 400 MHz              │
│ Typical Hash Rate: 500-600 GH/s         │
│ Power Draw:        40-50 W              │
│ Operating Temp:    50-70°C (optimal)    │
└─────────────────────────────────────────┘
```

---

## Support & Resources

**Official Firmware**:
- GitHub: https://github.com/skot/ESP-Miner
- Docs: https://esp-miner.readthedocs.io/

**Community**:
- Discord: ESP-Miner Community
- Reddit: r/BitcoinMining

**Hardware Datasheets**:
- BM1387: [Datasheet PDF](https://datasheet.lcsc.com/lcsc/1810171211_Bitmain-BM1387_C190127.pdf)
- ESP32: https://www.espressif.com/en/products/socs/esp32

---

**Document Version**: 1.0
**Last Updated**: 2025-12-18
**Maintainer**: CHIMERA Project Team
