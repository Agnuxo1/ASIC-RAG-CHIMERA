# LV06 Quick Start Guide - 5 Minutes to Mining

**Goal**: Get your LV06 mining to CHIMERA bridge in under 5 minutes.

---

## Prerequisites Checklist

- [ ] LV06 powered on (LED lit)
- [ ] Connected to same WiFi as your PC
- [ ] Know your PC's IP (example: `192.168.0.14`)
- [ ] Know LV06's IP (example: `192.168.0.15`)

**Don't know the IPs?** See [Finding IP Addresses](#finding-ip-addresses) below.

---

## Step 1: Configure Pool (2 minutes)

### Option A: Web Interface (Easiest)

1. Open browser: `http://192.168.0.15` (replace with your LV06 IP)
2. Click **"Pool Settings"** or **"Mining"**
3. Set Pool 1:
   ```
   URL:      stratum+tcp://192.168.0.14:3333
   User:     chimera
   Password: x
   ```
   (Replace `192.168.0.14` with YOUR PC's IP)
4. Click **Save**
5. Miner will restart automatically

### Option B: HTTP API (Advanced)

```bash
curl -X PATCH http://192.168.0.15/api/pool \
  -H "Content-Type: application/json" \
  -d '{
    "url": "stratum+tcp://192.168.0.14:3333",
    "user": "chimera",
    "pass": "x"
  }'

# Restart to apply
curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Step 2: Start CHIMERA Bridge (1 minute)

```bash
# Navigate to V04
cd "D:\Holographic Reservoir Computing\V04"

# Start bridge
python drivers/chronos_bridge.py
```

**Expected Output:**
```
⏳ CHRONOS LISTENER OPENED on 0.0.0.0:3333
🔗 API LISTENING on 4029
📊 TELEMETRY ENGINE STARTED
```

**Wait ~30 seconds** for miner to connect:
```
⚡ ASIC CONNECTED: 192.168.0.15
⚡ SENDING WAKE-UP SIGNAL: 400MHz
```

**Shares arriving:**
```
..........
📊 RITMO: CV=0.8523 | Entropía Temporal=2.1456
```

✅ **Success!** If you see dots and CV values, skip to [Step 3](#step-3-run-experiments-optional).

---

## Step 3: Run Experiments (Optional)

**Open a NEW terminal** (keep bridge running):

```bash
cd "D:\Holographic Reservoir Computing\V04"
python experiments/exp_01_voltage_modulation.py
```

This will run for ~12 minutes and generate a report.

---

## Finding IP Addresses

### Find Your PC's IP

**Windows:**
```cmd
ipconfig | findstr "IPv4"
```

**Linux/Mac:**
```bash
ip addr show | grep inet
# Or
ifconfig | grep inet
```

Look for something like: `192.168.0.14` or `192.168.1.100`

### Find LV06's IP

**Method 1: Router Admin Panel**
1. Open browser: `http://192.168.0.1` (or `192.168.1.1`)
2. Login (usually `admin` / `admin`)
3. Look for "Connected Devices" or "DHCP Clients"
4. Find device named `LuckyMiner` or `AxeOS`

**Method 2: Network Scan**
```bash
# Windows
arp -a | findstr "192.168"

# Linux
nmap -sn 192.168.0.0/24
```

**Method 3: Try Common IPs**
- `http://192.168.0.15`
- `http://192.168.0.100`
- `http://192.168.1.15`
- `http://192.168.4.1` (AP mode)

---

## Troubleshooting (Quick Fixes)

### Problem: Bridge shows no connection

**Check 1: Is pool URL correct?**
```bash
curl http://192.168.0.15/api/system/info
```
Look at pool URL. Should show your PC's IP.

**Check 2: Is bridge port 3333 open?**
```bash
netstat -an | findstr ":3333"
# Should show: 0.0.0.0:3333 LISTENING
```

**Fix**: Restart bridge and miner.

### Problem: CV = 0.0000

**Cause**: No shares arriving yet.

**Fix**: Wait 60 seconds. If still 0:
1. Check miner is hashing: `curl http://192.168.0.15/api/system/info` → `hashRate` should be >0
2. Restart miner pool config (Step 1)

### Problem: Miner too hot (>80°C)

**Immediate Fix**:
```bash
# Reduce frequency
curl -X PATCH http://192.168.0.15/api/system \
  -H "Content-Type: application/json" \
  -d '{"frequency": 300, "volts": 950}'

# Restart
curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

Check fan is spinning.

---

## Default Settings

If you mess up config, reset to these safe defaults:

| Setting | Value |
|---------|-------|
| Voltage | 990 mV |
| Frequency | 400 MHz |
| Pool URL | `stratum+tcp://192.168.0.14:3333` |
| Pool User | `chimera` |
| Pool Pass | `x` |

**HTTP API Reset**:
```bash
curl -X PATCH http://192.168.0.15/api/system \
  -H "Content-Type: application/json" \
  -d '{"frequency": 400, "volts": 990}'

curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Next Steps

✅ **You're now mining to CHIMERA!**

- Check real-time metrics: `curl http://127.0.0.1:4029/GET_METRICS` (wait, wrong command)
- Run voltage experiments: `python experiments/exp_01_voltage_modulation.py`
- Read full docs: `README.md`

---

## Emergency Contacts

**Hardware Issue**: Check heat sink, fan, power supply
**Network Issue**: Verify IPs with `ipconfig` / `arp -a`
**Software Issue**: Restart bridge, restart miner, reboot PC

**Factory Reset**: Hold button on LV06 for 10 seconds → Reconfigure from `192.168.4.1`

---

**Time to Success**: ~5 minutes
**Difficulty**: ⭐⭐☆☆☆ (Easy)
**Prerequisites**: Basic command line knowledge
