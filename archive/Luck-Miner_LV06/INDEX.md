# Lucky Miner LV06 - Documentation Index

**Complete documentation for configuring and using Lucky Miner LV06 with CHIMERA system**

---

## 📚 Documentation Structure

### 🚀 Quick Start (For Beginners)

**Start here if you want to get up and running fast:**

| Document | Description | Time Required |
|----------|-------------|---------------|
| [QUICK_START.md](QUICK_START.md) | 5-minute setup guide | 5 minutes |
| [test_connection.py](test_connection.py) | Connection testing script | 2 minutes |

**Recommended Path**: QUICK_START.md → test_connection.py → Start experimenting

---

### 📖 Complete Reference (For IT Professionals)

**Comprehensive documentation for full understanding:**

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | Complete hardware & software guide | System Administrators |
| [API_EXAMPLES.md](API_EXAMPLES.md) | Code examples (Python, curl, PowerShell, JavaScript) | Developers |

---

### 🛠️ Tools & Scripts

**Automated configuration and testing utilities:**

| Script | Purpose | Usage |
|--------|---------|-------|
| [test_connection.py](test_connection.py) | Test LV06 connectivity & status | `python test_connection.py 192.168.0.15` |
| [configure_lv06.py](configure_lv06.py) | Auto-configure LV06 for CHIMERA | `python configure_lv06.py --miner-ip 192.168.0.15 --auto` |

---

## 🎯 Quick Navigation by Task

### I want to...

#### ...Get Started Quickly
→ Read [QUICK_START.md](QUICK_START.md) (5 minutes)

#### ...Understand the Hardware
→ Read [README.md - Hardware Overview](README.md#hardware-overview)

#### ...Configure the Miner
→ Read [README.md - Pool Configuration](README.md#pool-configuration)
→ Or use [configure_lv06.py](configure_lv06.py) for auto-config

#### ...Test if it's Working
→ Run [test_connection.py](test_connection.py)

#### ...Write Custom Code
→ Read [API_EXAMPLES.md](API_EXAMPLES.md)

#### ...Troubleshoot Problems
→ Read [README.md - Troubleshooting](README.md#troubleshooting)
→ Read [QUICK_START.md - Troubleshooting](QUICK_START.md#troubleshooting-quick-fixes)

#### ...Change Voltage/Frequency
→ Read [README.md - Advanced Configuration](README.md#advanced-configuration)
→ Read [API_EXAMPLES.md - Voltage Change](API_EXAMPLES.md#2-set-voltage-and-frequency)

#### ...Monitor Temperature
→ Read [API_EXAMPLES.md - Monitor Temperature](API_EXAMPLES.md#3-monitor-temperature-in-real-time)

#### ...Run CHIMERA Experiments
→ Configure pool to PC's IP (QUICK_START.md)
→ Start bridge: `python V04/drivers/chronos_bridge.py`
→ Run experiment: `python V04/experiments/exp_01_voltage_modulation.py`

---

## 📋 Prerequisites

### Hardware
- Lucky Miner LV06 (BM1387 ASIC)
- 5V or 12V power supply (check your model)
- WiFi network

### Software
- Python 3.7+ with `requests` library
- Network access to LV06

### Network
- LV06 and PC on same network
- Know both IP addresses

---

## 🔍 Document Details

### QUICK_START.md
**Purpose**: Get mining in 5 minutes
**Length**: ~5 pages
**Includes**:
- Step-by-step pool configuration
- CHIMERA bridge startup
- Quick troubleshooting

**Best For**: First-time users, quick reference

---

### README.md
**Purpose**: Complete reference manual
**Length**: ~30 pages
**Includes**:
- Hardware specifications
- Network configuration
- AxeOS HTTP API reference
- Advanced configuration
- Safety warnings
- Troubleshooting guide

**Best For**: System administrators, IT professionals, advanced users

---

### API_EXAMPLES.md
**Purpose**: Programming reference with code examples
**Length**: ~25 pages
**Includes**:
- Python examples (requests library)
- curl examples (Linux/Mac)
- PowerShell examples (Windows)
- JavaScript examples (Node.js & Browser)
- Common patterns (error handling, safe voltage change)

**Best For**: Developers, automation engineers, scripters

---

### test_connection.py
**Purpose**: Automated connectivity testing
**Type**: Python script
**Features**:
- Network ping test
- HTTP API validation
- Pool config check
- Hashing status verification
- CHIMERA bridge compatibility test

**Usage**:
```bash
python test_connection.py 192.168.0.15
```

**Output**: Pass/Fail report with troubleshooting tips

---

### configure_lv06.py
**Purpose**: Automated LV06 configuration
**Type**: Python script
**Features**:
- Auto-detects PC IP
- Configures pool settings
- Sets voltage/frequency
- Restarts miner
- Verifies configuration

**Usage**:
```bash
# Auto-detect PC IP
python configure_lv06.py --miner-ip 192.168.0.15 --auto

# Manual PC IP
python configure_lv06.py --miner-ip 192.168.0.15 --pc-ip 192.168.0.14

# Custom voltage/frequency
python configure_lv06.py --miner-ip 192.168.0.15 --auto --voltage 950 --frequency 350
```

---

## 🎓 Learning Path

### Beginner (30 minutes)
1. Read QUICK_START.md (5 min)
2. Run test_connection.py (2 min)
3. Configure pool via web UI (5 min)
4. Start CHIMERA bridge (2 min)
5. Verify connection (5 min)
6. Experiment with basic commands (11 min)

### Intermediate (2 hours)
1. Complete beginner path
2. Read README.md sections:
   - Hardware Overview (15 min)
   - Network Configuration (15 min)
   - AxeOS HTTP API Reference (30 min)
3. Run configure_lv06.py script (10 min)
4. Try API examples in API_EXAMPLES.md (30 min)
5. Run voltage modulation experiment (20 min)

### Advanced (4+ hours)
1. Complete intermediate path
2. Read all documentation thoroughly
3. Experiment with voltage/frequency optimization
4. Write custom monitoring scripts
5. Set up Prometheus/Grafana dashboard (optional)
6. Contribute improvements to CHIMERA project

---

## 📊 Specifications Quick Reference

```
┌─────────────────────────────────────────┐
│  Lucky Miner LV06 - Specifications      │
├─────────────────────────────────────────┤
│ ASIC Chip:         BM1387               │
│ Hash Rate:         500-600 GH/s         │
│ Power:             40-50W               │
│ Voltage Range:     850-1200 mV          │
│ Frequency Range:   50-600 MHz           │
│ Network:           WiFi 802.11 b/g/n    │
│ Default IP:        192.168.0.15 (DHCP)  │
│ Web Interface:     http://<ip>          │
│ API Port:          80                   │
│ Stratum Port:      3333                 │
│ Firmware:          AxeOS (ESP-Miner)    │
└─────────────────────────────────────────┘
```

---

## ⚠️ Safety Reminders

**DO NOT**:
- Exceed 1200 mV voltage
- Run above 85°C for extended periods
- Block airflow to heat sink
- Touch ASIC while powered (burn hazard)

**DO**:
- Monitor temperature regularly
- Start with defaults (990 mV, 400 MHz)
- Test voltage changes incrementally
- Keep firmware updated

---

## 🔗 Related Resources

### CHIMERA Project
- V03 Documentation: `../V03/README_V3.md`
- V04 Guide: `../V04/GUIA_EJECUCION_V04.md`
- Corrections: `../V04/CORRECCIONES_V04.md`

### External Resources
- AxeOS GitHub: https://github.com/skot/ESP-Miner
- BM1387 Datasheet: [PDF Link]
- ESP32 Docs: https://www.espressif.com/

---

## 📞 Support

**Hardware Issues**:
- Check power supply (5V/12V @ 2A+)
- Verify heat sink and fan operation
- Measure voltage with multimeter if available

**Network Issues**:
- Verify IPs: `ipconfig` (Windows) or `ip addr` (Linux)
- Check router DHCP leases
- Try factory reset (hold button 10s)

**Software Issues**:
- Update AxeOS firmware
- Check Python dependencies: `pip install requests`
- Review logs in CHIMERA bridge

**No Response**:
- Try AP mode: Connect to `LuckyMiner-XXXX`, access `192.168.4.1`
- Serial console: 115200 baud, 8N1
- Power cycle (unplug, wait 10s, replug)

---

## 📝 Version History

**v1.0** (2025-12-18)
- Initial documentation release
- All scripts tested with LV06 + AxeOS v2.x
- Compatible with CHIMERA V04

---

## 👥 Contributors

- CHIMERA Project Team
- AxeOS Community
- ESP-Miner Contributors

---

**Last Updated**: 2025-12-18
**Maintained By**: CHIMERA Project
**License**: MIT (Scripts), CC-BY-4.0 (Documentation)
