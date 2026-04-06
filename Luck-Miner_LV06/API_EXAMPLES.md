# LV06 HTTP API - Code Examples

Complete examples for interacting with Lucky Miner LV06 via HTTP API.

---

## Table of Contents

1. [Python Examples](#python-examples)
2. [curl Examples](#curl-examples)
3. [PowerShell Examples](#powershell-examples)
4. [JavaScript Examples](#javascript-examples)

---

## Python Examples

### 1. Get System Info

```python
import requests
import json

MINER_IP = "192.168.0.15"

def get_system_info():
    url = f"http://{MINER_IP}/api/system/info"
    response = requests.get(url, timeout=5)

    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
info = get_system_info()
print(f"Temperature: {info['temp']}°C")
print(f"Hash Rate: {info['hashRate']} GH/s")
```

### 2. Set Voltage and Frequency

```python
import requests
import time

MINER_IP = "192.168.0.15"

def set_voltage_frequency(voltage_mv, frequency_mhz):
    """Set voltage and frequency, then restart"""

    # Step 1: Update settings
    url = f"http://{MINER_IP}/api/system"
    payload = {
        "frequency": int(frequency_mhz),
        "volts": int(voltage_mv)
    }

    response = requests.patch(url, json=payload, timeout=5)

    if response.status_code != 200:
        print(f"Config failed: {response.status_code}")
        return False

    print(f"✅ Config updated: {voltage_mv}mV @ {frequency_mhz}MHz")

    # Step 2: Restart miner
    restart_url = f"http://{MINER_IP}/api/system/restart"

    try:
        requests.post(restart_url, json={}, timeout=3)
    except requests.exceptions.Timeout:
        pass  # Expected - miner reboots immediately

    print("⏳ Restarting miner (40s)...")
    time.sleep(40)

    # Step 3: Verify
    info_url = f"http://{MINER_IP}/api/system/info"
    response = requests.get(info_url, timeout=5)

    if response.status_code == 200:
        data = response.json()
        actual_v = data.get('voltage', data.get('coreVoltageActual', 0))
        actual_f = data.get('frequency', 0)

        print(f"✅ Verified: {actual_v}mV @ {actual_f}MHz")
        return True
    else:
        print("⚠️  Could not verify (miner may still be booting)")
        return False

# Usage
set_voltage_frequency(950, 350)
```

### 3. Monitor Temperature in Real-Time

```python
import requests
import time
from datetime import datetime

MINER_IP = "192.168.0.15"
POLL_INTERVAL = 3  # seconds

def monitor_temperature(duration_minutes=10, temp_threshold=75):
    """Monitor temperature and alert if too high"""

    url = f"http://{MINER_IP}/api/system/info"
    end_time = time.time() + (duration_minutes * 60)

    print(f"Monitoring temperature for {duration_minutes} minutes...")
    print(f"Alert threshold: {temp_threshold}°C\n")

    while time.time() < end_time:
        try:
            response = requests.get(url, timeout=5)
            data = response.json()

            temp = data.get('temp', 0)
            hashrate = data.get('hashRate', 0)
            power = data.get('power', 0)

            timestamp = datetime.now().strftime('%H:%M:%S')

            status = "OK"
            if temp >= temp_threshold:
                status = "⚠️  HIGH"
            elif temp >= temp_threshold + 10:
                status = "🔥 CRITICAL"

            print(f"[{timestamp}] {temp:5.1f}°C | {hashrate:6.2f} GH/s | "
                  f"{power:5.1f}W | {status}")

            if temp >= temp_threshold + 10:
                print("   🚨 CRITICAL TEMPERATURE - Consider reducing frequency!")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(POLL_INTERVAL)

# Usage
monitor_temperature(duration_minutes=5, temp_threshold=75)
```

### 4. Voltage Sweep Experiment

```python
import requests
import time
import json

MINER_IP = "192.168.0.15"

def voltage_sweep(voltages=[900, 950, 990, 1050], stabilize_time=60):
    """Test different voltages and record metrics"""

    results = []

    for voltage in voltages:
        print(f"\n--- Testing {voltage}mV ---")

        # Set voltage
        url = f"http://{MINER_IP}/api/system"
        payload = {"frequency": 400, "volts": voltage}
        requests.patch(url, json=payload)

        # Restart
        restart_url = f"http://{MINER_IP}/api/system/restart"
        try:
            requests.post(restart_url, json={}, timeout=3)
        except:
            pass

        print(f"⏳ Waiting {stabilize_time}s for stabilization...")
        time.sleep(stabilize_time)

        # Collect metrics (average over 5 samples)
        samples = []
        for i in range(5):
            info_url = f"http://{MINER_IP}/api/system/info"
            response = requests.get(info_url)
            data = response.json()
            samples.append(data)
            time.sleep(2)

        # Calculate averages
        avg_temp = sum(s.get('temp', 0) for s in samples) / len(samples)
        avg_hashrate = sum(s.get('hashRate', 0) for s in samples) / len(samples)
        avg_power = sum(s.get('power', 0) for s in samples) / len(samples)

        efficiency = avg_hashrate / avg_power if avg_power > 0 else 0

        result = {
            "voltage": voltage,
            "temperature": round(avg_temp, 1),
            "hashrate": round(avg_hashrate, 2),
            "power": round(avg_power, 2),
            "efficiency": round(efficiency, 2)
        }

        results.append(result)

        print(f"   Temp: {result['temperature']}°C")
        print(f"   Hash: {result['hashrate']} GH/s")
        print(f"   Power: {result['power']}W")
        print(f"   Efficiency: {result['efficiency']} GH/W")

    # Save results
    with open('voltage_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to: voltage_sweep_results.json")

    return results

# Usage
results = voltage_sweep()
```

---

## curl Examples

### 1. Get System Info

```bash
curl http://192.168.0.15/api/system/info
```

**With formatting (requires jq):**
```bash
curl -s http://192.168.0.15/api/system/info | jq
```

### 2. Set Voltage and Frequency

```bash
curl -X PATCH http://192.168.0.15/api/system \
  -H "Content-Type: application/json" \
  -d '{"frequency": 400, "volts": 990}'
```

### 3. Restart Miner

```bash
curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 4. Monitor in Loop (Linux/Mac)

```bash
#!/bin/bash
while true; do
  echo "=== $(date) ==="
  curl -s http://192.168.0.15/api/system/info | jq '.temp, .hashRate, .power'
  sleep 5
done
```

---

## PowerShell Examples

### 1. Get System Info

```powershell
$MinerIP = "192.168.0.15"
$Url = "http://$MinerIP/api/system/info"

$Response = Invoke-RestMethod -Uri $Url -Method Get
$Response | ConvertTo-Json

Write-Host "Temperature: $($Response.temp)°C"
Write-Host "Hash Rate: $($Response.hashRate) GH/s"
Write-Host "Power: $($Response.power)W"
```

### 2. Set Voltage and Frequency

```powershell
$MinerIP = "192.168.0.15"
$Url = "http://$MinerIP/api/system"

$Body = @{
    frequency = 400
    volts = 990
} | ConvertTo-Json

Invoke-RestMethod -Uri $Url -Method Patch -Body $Body -ContentType "application/json"
Write-Host "✅ Configuration updated"

# Restart
$RestartUrl = "http://$MinerIP/api/system/restart"
Invoke-RestMethod -Uri $RestartUrl -Method Post -Body "{}" -ContentType "application/json"
Write-Host "⏳ Restarting miner..."

Start-Sleep -Seconds 40
Write-Host "✅ Reboot complete"
```

### 3. Monitor Temperature

```powershell
$MinerIP = "192.168.0.15"
$Duration = 300  # 5 minutes
$Interval = 5    # 5 seconds

$EndTime = (Get-Date).AddSeconds($Duration)

while ((Get-Date) -lt $EndTime) {
    $Response = Invoke-RestMethod -Uri "http://$MinerIP/api/system/info" -Method Get

    $Timestamp = Get-Date -Format "HH:mm:ss"
    $Temp = $Response.temp
    $HashRate = $Response.hashRate
    $Power = $Response.power

    Write-Host "[$Timestamp] $Temp°C | $HashRate GH/s | $($Power)W"

    if ($Temp -gt 80) {
        Write-Host "   ⚠️  HIGH TEMPERATURE!" -ForegroundColor Red
    }

    Start-Sleep -Seconds $Interval
}
```

---

## JavaScript Examples

### 1. Get System Info (Node.js)

```javascript
const http = require('http');

const MINER_IP = '192.168.0.15';

function getSystemInfo() {
    return new Promise((resolve, reject) => {
        const url = `http://${MINER_IP}/api/system/info`;

        http.get(url, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                resolve(JSON.parse(data));
            });
        }).on('error', reject);
    });
}

// Usage
getSystemInfo().then(info => {
    console.log(JSON.stringify(info, null, 2));
    console.log(`Temperature: ${info.temp}°C`);
    console.log(`Hash Rate: ${info.hashRate} GH/s`);
});
```

### 2. Set Voltage (Using axios)

```javascript
const axios = require('axios');

const MINER_IP = '192.168.0.15';

async function setVoltageFrequency(voltage, frequency) {
    try {
        // Update config
        const configUrl = `http://${MINER_IP}/api/system`;
        await axios.patch(configUrl, {
            frequency: frequency,
            volts: voltage
        });

        console.log(`✅ Config updated: ${voltage}mV @ ${frequency}MHz`);

        // Restart
        const restartUrl = `http://${MINER_IP}/api/system/restart`;
        await axios.post(restartUrl, {});

        console.log('⏳ Restarting miner (40s)...');
        await new Promise(resolve => setTimeout(resolve, 40000));

        // Verify
        const infoUrl = `http://${MINER_IP}/api/system/info`;
        const response = await axios.get(infoUrl);

        console.log(`✅ Verified: ${response.data.voltage}mV @ ${response.data.frequency}MHz`);

    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Usage
setVoltageFrequency(950, 350);
```

### 3. Web Dashboard (Browser)

```html
<!DOCTYPE html>
<html>
<head>
    <title>LV06 Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>LV06 Miner Dashboard</h1>
    <div id="status"></div>
    <canvas id="chart" width="400" height="200"></canvas>

    <script>
        const MINER_IP = '192.168.0.15';
        const tempData = [];
        const timeLabels = [];

        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: tempData,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            }
        });

        async function updateStatus() {
            try {
                const response = await fetch(`http://${MINER_IP}/api/system/info`);
                const data = await response.json();

                // Update display
                document.getElementById('status').innerHTML = `
                    <p>Temperature: ${data.temp}°C</p>
                    <p>Hash Rate: ${data.hashRate} GH/s</p>
                    <p>Power: ${data.power}W</p>
                    <p>Voltage: ${data.voltage}mV</p>
                `;

                // Update chart
                const now = new Date().toLocaleTimeString();
                timeLabels.push(now);
                tempData.push(data.temp);

                if (tempData.length > 20) {
                    tempData.shift();
                    timeLabels.shift();
                }

                chart.update();

            } catch (error) {
                console.error('Error:', error);
            }
        }

        // Update every 5 seconds
        setInterval(updateStatus, 5000);
        updateStatus();  // Initial update
    </script>
</body>
</html>
```

---

## Common Patterns

### Safe Voltage Change

Always follow this pattern to avoid damage:

```python
def safe_voltage_change(current_voltage, target_voltage, step=50):
    """Change voltage in small steps"""

    if target_voltage > current_voltage:
        voltages = range(current_voltage, target_voltage + 1, step)
    else:
        voltages = range(current_voltage, target_voltage - 1, -step)

    for voltage in voltages:
        print(f"Setting {voltage}mV...")
        set_voltage_frequency(voltage, 400)
        time.sleep(30)  # Stabilize

        # Check temperature
        info = get_system_info()
        if info['temp'] > 80:
            print("⚠️  Too hot! Reverting...")
            set_voltage_frequency(current_voltage, 400)
            return False

    return True
```

### Error Handling

Always handle network errors:

```python
import requests
from requests.exceptions import RequestException

def robust_api_call(url, retries=3, timeout=5):
    """API call with retry logic"""

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"   Retry {attempt + 1}/{retries}...")
            time.sleep(2)
```

---

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/system/info` | Get system status |
| PATCH | `/api/system` | Update voltage/frequency |
| POST | `/api/system/restart` | Restart miner |
| GET | `/api/system/wifi` | Get WiFi info |
| GET | `/api/pool` | Get pool config (varies) |
| PATCH | `/api/pool` | Update pool (varies) |

**Note**: Exact endpoints may vary by firmware version. Check web UI network tab for your specific version.

---

**Last Updated**: 2025-12-18
**Tested Firmware**: AxeOS v2.x
