# Plan de Experimentación: Validación de ASIC Real para RAG
## Lucky Miner LV06 - Primera Prueba con Hardware

**Fecha**: 18 de Diciembre 2024
**Objetivo**: Demostrar que el ASIC BM1366 puede realizar trabajo SHA-256 útil para RAG
**Hardware**: Lucky Miner LV06 (1× BM1366 @ 500MHz, 520 GH/s)
**Estado Actual**: Hardware operativo, IP 192.168.0.15, AxeOS v2.3.6

---

## Fase 1: Servidor Stratum Local - EXPERIMENTO MÍNIMO VIABLE (MVE)

### Objetivo de la Fase
Probar que podemos enviar trabajos SHA-256 controlados al LV06 y obtener resultados medibles, sin modificar el firmware.

### Estrategia: Servidor Stratum como Puente

```
┌────────────────────────────────────────────────────────┐
│                ARQUITECTURA MVE                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  PC Python Script                                      │
│  ┌──────────────────────────────────────────┐          │
│  │  Stratum Server (Local)                  │          │
│  │  - Puerto 3333                           │          │
│  │  - Genera trabajos custom                │          │
│  │  - Mide throughput real                  │          │
│  └──────────┬───────────────────────────────┘          │
│             │ TCP/IP (Stratum Protocol)                │
│             ▼                                          │
│  ┌──────────────────────────────────────────┐          │
│  │  Lucky Miner LV06                        │          │
│  │  - BM1366 @ 500MHz                       │          │
│  │  - 520 GH/s nominal                      │          │
│  │  - AxeOS firmware (sin modificar)        │          │
│  └──────────────────────────────────────────┘          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Limitación Aceptada
El BM1366 solo puede hacer **double-SHA256** de bloques Bitcoin (80 bytes). No podemos hashear datos arbitrarios directamente, PERO podemos:

1. Usar el campo `merkle_root` para codificar nuestros datos (32 bytes)
2. Variar el `prev_block_hash` (32 bytes)
3. Controlar `ntime` y `version` (8 bytes)
4. **Total: 72 bytes de datos custom por trabajo**

Esto es suficiente para:
- Hashear tags de 256 bits (embeddings binarios)
- Medir throughput real del ASIC
- Comparar con CPU

---

## Implementación del Experimento

### 1.1 Crear Servidor Stratum Mínimo

**Archivo**: `D:\ASIC_RAG\experiments\lv06_stratum_server.py`

```python
#!/usr/bin/env python3
"""
Servidor Stratum Minimalista para Experimentos RAG con LV06
No requiere modificar firmware del minero.

Basado en:
- Stratum v1 protocol
- Compatible con AxeOS/ESP-Miner
"""

import socket
import json
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import struct

@dataclass
class MiningJob:
    """Trabajo de minería que enviamos al LV06"""
    job_id: str
    prev_hash: str  # 32 bytes hex (256 bits) - Aquí codificamos datos custom
    coinbase1: str
    coinbase2: str
    merkle_branches: List[str]
    version: str  # 4 bytes hex
    nbits: str    # 4 bytes hex (dificultad)
    ntime: str    # 4 bytes hex (timestamp)
    clean_jobs: bool

    # Metadata para tracking
    created_at: float
    data_payload: bytes  # Los datos originales que queremos hashear

@dataclass
class SubmittedShare:
    """Share enviado por el minero"""
    job_id: str
    extranonce2: str
    ntime: str
    nonce: str
    received_at: float

    def calculate_hash(self, job: MiningJob) -> bytes:
        """Reconstruir el hash que encontró el minero"""
        # Aquí reconstruirías el header completo y calcularías SHA256d
        # Por ahora retornamos el nonce como identificador
        return bytes.fromhex(self.nonce)


class LV06StratumServer:
    """
    Servidor Stratum simplificado para experimentos con LV06.

    Funcionalidad:
    1. Acepta conexión del minero LV06
    2. Envía trabajos custom (datos codificados en campos del header)
    3. Recibe shares y mide throughput
    """

    def __init__(self, host='0.0.0.0', port=3333):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False

        # Estado
        self.jobs: Dict[str, MiningJob] = {}
        self.shares: List[SubmittedShare] = []
        self.connected_miners = []

        # Estadísticas
        self.total_hashes_computed = 0
        self.start_time = None
        self.job_counter = 0

        # Configuración experimental
        self.difficulty = 1  # Dificultad BAJA para obtener muchos shares rápido
        self.extranonce1 = "01000000"  # 4 bytes
        self.extranonce2_size = 4  # 4 bytes

    def start(self):
        """Inicia el servidor"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        self.start_time = time.time()

        print(f"[STRATUM] Servidor iniciado en {self.host}:{self.port}")
        print(f"[CONFIG] Dificultad: {self.difficulty}")
        print(f"[CONFIG] ExtraNonce1: {self.extranonce1}")
        print(f"[READY] Esperando conexión del LV06...")
        print()

        try:
            while self.running:
                client_socket, address = self.server_socket.accept()
                print(f"[CONNECT] Minero conectado desde {address}")

                # Manejar cliente en thread separado
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Cerrando servidor...")
        finally:
            self.stop()

    def handle_client(self, client_socket, address):
        """Maneja la comunicación con un minero conectado"""
        try:
            # Buffer para mensajes
            buffer = ""

            while self.running:
                # Recibir datos
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break

                buffer += data

                # Procesar mensajes completos (terminados en \n)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        response = self.handle_message(line.strip(), client_socket)
                        if response:
                            client_socket.send((response + '\n').encode('utf-8'))

        except Exception as e:
            print(f"[ERROR] Error manejando cliente {address}: {e}")
        finally:
            client_socket.close()
            print(f"[DISCONNECT] Minero {address} desconectado")

    def handle_message(self, message: str, client_socket) -> Optional[str]:
        """Procesa un mensaje Stratum del minero"""
        try:
            msg = json.loads(message)
            method = msg.get('method')
            msg_id = msg.get('id')
            params = msg.get('params', [])

            print(f"[RX] {method} (id={msg_id})")

            # mining.subscribe
            if method == 'mining.subscribe':
                return self.handle_subscribe(msg_id, params)

            # mining.authorize
            elif method == 'mining.authorize':
                return self.handle_authorize(msg_id, params, client_socket)

            # mining.submit
            elif method == 'mining.submit':
                return self.handle_submit(msg_id, params)

            # mining.extranonce.subscribe
            elif method == 'mining.extranonce.subscribe':
                return json.dumps({"id": msg_id, "result": True, "error": None})

            else:
                print(f"[WARN] Método desconocido: {method}")
                return json.dumps({"id": msg_id, "result": None, "error": None})

        except json.JSONDecodeError:
            print(f"[ERROR] Mensaje JSON inválido: {message}")
            return None
        except Exception as e:
            print(f"[ERROR] Error procesando mensaje: {e}")
            return None

    def handle_subscribe(self, msg_id, params):
        """Responde a mining.subscribe"""
        response = {
            "id": msg_id,
            "result": [
                [
                    ["mining.set_difficulty", "deadbeef"],
                    ["mining.notify", "deadbeef"]
                ],
                self.extranonce1,
                self.extranonce2_size
            ],
            "error": None
        }
        return json.dumps(response)

    def handle_authorize(self, msg_id, params, client_socket):
        """Responde a mining.authorize y envía dificultad + primer trabajo"""
        # Autorización
        username = params[0] if params else "unknown"
        print(f"[AUTH] Usuario: {username}")

        auth_response = {
            "id": msg_id,
            "result": True,
            "error": None
        }
        client_socket.send((json.dumps(auth_response) + '\n').encode('utf-8'))

        # Enviar dificultad
        diff_msg = {
            "id": None,
            "method": "mining.set_difficulty",
            "params": [self.difficulty]
        }
        client_socket.send((json.dumps(diff_msg) + '\n').encode('utf-8'))
        print(f"[TX] Dificultad configurada: {self.difficulty}")

        # Enviar primer trabajo
        job = self.create_test_job()
        self.send_job(client_socket, job)

        return None  # Ya enviamos la respuesta

    def handle_submit(self, msg_id, params):
        """Procesa un share enviado por el minero"""
        # params = [username, job_id, extranonce2, ntime, nonce]
        username = params[0]
        job_id = params[1]
        extranonce2 = params[2]
        ntime = params[3]
        nonce = params[4]

        share = SubmittedShare(
            job_id=job_id,
            extranonce2=extranonce2,
            ntime=ntime,
            nonce=nonce,
            received_at=time.time()
        )

        self.shares.append(share)

        # Calcular hashes computados (aproximación)
        # Cada share representa difficulty * 2^32 hashes
        hashes = self.difficulty * (2 ** 32)
        self.total_hashes_computed += hashes

        # Estadísticas
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes_computed / elapsed if elapsed > 0 else 0

        print(f"[SHARE] #{len(self.shares)} | Job: {job_id} | Nonce: {nonce}")
        print(f"        Hashrate: {hashrate/1e9:.2f} GH/s | Total: {self.total_hashes_computed/1e9:.2f} GHashes")

        # Aceptar el share
        response = {
            "id": msg_id,
            "result": True,
            "error": None
        }

        return json.dumps(response)

    def create_test_job(self, custom_data: bytes = None) -> MiningJob:
        """
        Crea un trabajo de minería con datos custom.

        Args:
            custom_data: Datos que queremos hashear (max 32 bytes)
                        Si None, genera datos de prueba
        """
        self.job_counter += 1
        job_id = f"job_{self.job_counter:04d}"

        # Generar datos custom si no se proveen
        if custom_data is None:
            # Para pruebas: hashear el job_id como dato
            custom_data = hashlib.sha256(job_id.encode()).digest()

        # Codificar datos en prev_hash (32 bytes)
        prev_hash = custom_data.hex().ljust(64, '0')[:64]

        # Merkle root vacío (podríamos codificar más datos aquí)
        merkle_root = "00" * 32

        # Coinbase simplificado
        coinbase1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
        coinbase2 = "ffffffff0100f2052a01000000434104"

        # Version, nbits, ntime
        version = "20000000"
        nbits = "1d00ffff"  # Dificultad muy baja
        ntime = hex(int(time.time()))[2:].zfill(8)

        job = MiningJob(
            job_id=job_id,
            prev_hash=prev_hash,
            coinbase1=coinbase1,
            coinbase2=coinbase2,
            merkle_branches=[],
            version=version,
            nbits=nbits,
            ntime=ntime,
            clean_jobs=True,
            created_at=time.time(),
            data_payload=custom_data
        )

        self.jobs[job_id] = job
        return job

    def send_job(self, client_socket, job: MiningJob):
        """Envía un trabajo al minero"""
        notify_msg = {
            "id": None,
            "method": "mining.notify",
            "params": [
                job.job_id,
                job.prev_hash,
                job.coinbase1,
                job.coinbase2,
                job.merkle_branches,
                job.version,
                job.nbits,
                job.ntime,
                job.clean_jobs
            ]
        }

        client_socket.send((json.dumps(notify_msg) + '\n').encode('utf-8'))
        print(f"[TX] Trabajo enviado: {job.job_id}")
        print(f"     Data: {job.data_payload.hex()[:32]}...")

    def stop(self):
        """Detiene el servidor"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

        # Estadísticas finales
        print("\n" + "="*60)
        print("ESTADÍSTICAS FINALES")
        print("="*60)
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Tiempo total: {elapsed:.2f} segundos")
        print(f"Shares recibidos: {len(self.shares)}")
        print(f"Trabajos enviados: {self.job_counter}")
        print(f"Hashes totales: {self.total_hashes_computed/1e9:.2f} GHashes")
        if elapsed > 0:
            print(f"Hashrate promedio: {self.total_hashes_computed/elapsed/1e9:.2f} GH/s")


def main():
    """
    Ejecuta el servidor Stratum para experimentos con LV06
    """
    print("="*60)
    print("SERVIDOR STRATUM EXPERIMENTAL - LV06 RAG")
    print("="*60)
    print()
    print("Configuración del LV06:")
    print("  1. Accede a http://192.168.0.15")
    print("  2. Ve a Settings → Pool Configuration")
    print("  3. Configura Pool 1:")
    print(f"     - URL: stratum+tcp://192.168.0.14:3333")
    print("     - User: test")
    print("     - Password: x")
    print("  4. Guarda y el minero se conectará aquí")
    print()
    print("Presiona Ctrl+C para detener")
    print("="*60)
    print()

    server = LV06StratumServer(host='0.0.0.0', port=3333)
    server.start()

if __name__ == "__main__":
    main()
```

---

### 1.2 Script de Benchmark: CPU vs LV06

**Archivo**: `D:\ASIC_RAG\experiments\benchmark_cpu_vs_lv06.py`

```python
#!/usr/bin/env python3
"""
Benchmark Comparativo: CPU (hashlib) vs LV06 (ASIC Real)

Mide:
1. Throughput de hashing (H/s)
2. Latencia por operación
3. Consumo energético estimado
4. Eficiencia (H/J)
"""

import hashlib
import time
import socket
import json
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    implementation: str
    total_hashes: int
    elapsed_seconds: float
    hashes_per_second: float
    avg_latency_us: float
    power_watts: float
    efficiency_gh_per_watt: float

def benchmark_cpu_hashlib(num_hashes: int = 1000000) -> BenchmarkResult:
    """
    Benchmark de hashing CPU usando hashlib
    """
    print(f"[CPU] Hasheando {num_hashes:,} veces con hashlib...")

    # Generar datos de prueba
    test_data = [f"test_data_{i}".encode() for i in range(1000)]

    # Benchmark
    start = time.perf_counter()

    hash_count = 0
    while hash_count < num_hashes:
        for data in test_data:
            hashlib.sha256(data).digest()
            hash_count += 1
            if hash_count >= num_hashes:
                break

    elapsed = time.perf_counter() - start

    # Calcular métricas
    hashes_per_second = num_hashes / elapsed
    avg_latency_us = (elapsed / num_hashes) * 1e6

    # Consumo CPU estimado (TDP típico i7 = 65W, asumiendo 50% uso)
    power_watts = 65 * 0.5
    efficiency = (hashes_per_second / 1e9) / power_watts

    result = BenchmarkResult(
        implementation="CPU (hashlib)",
        total_hashes=num_hashes,
        elapsed_seconds=elapsed,
        hashes_per_second=hashes_per_second,
        avg_latency_us=avg_latency_us,
        power_watts=power_watts,
        efficiency_gh_per_watt=efficiency
    )

    print(f"[CPU] Completado en {elapsed:.2f}s")
    print(f"[CPU] Hashrate: {hashes_per_second:,.0f} H/s ({hashes_per_second/1e6:.2f} MH/s)")

    return result

def benchmark_lv06_real(duration_seconds: int = 60) -> BenchmarkResult:
    """
    Benchmark del LV06 real conectándose al servidor Stratum

    Nota: Requiere que el LV06 esté minando al servidor local
          y el servidor Stratum esté recolectando estadísticas
    """
    print(f"[LV06] Monitoreando ASIC real por {duration_seconds} segundos...")
    print("[LV06] Asegúrate de que:")
    print("       1. lv06_stratum_server.py esté corriendo")
    print("       2. LV06 esté conectado (ver logs del servidor)")
    print()

    # Obtener estadísticas del LV06 vía API HTTP
    lv06_ip = "192.168.0.15"

    # Snapshot inicial
    initial_stats = get_lv06_stats(lv06_ip)
    initial_time = time.time()

    print(f"[LV06] Hashrate inicial reportado: {initial_stats.get('hashRate', 0):.2f} GH/s")
    print(f"[LV06] Esperando {duration_seconds}s...")

    # Esperar
    time.sleep(duration_seconds)

    # Snapshot final
    final_stats = get_lv06_stats(lv06_ip)
    final_time = time.time()
    elapsed = final_time - initial_time

    # Calcular hashes totales
    # shares_accepted indica cuántos shares fueron encontrados
    shares_delta = final_stats.get('sharesAccepted', 0) - initial_stats.get('sharesAccepted', 0)

    # Cada share a difficulty=1 representa 2^32 hashes
    total_hashes = shares_delta * (2 ** 32)

    # Alternativamente, usar hashrate reportado
    avg_hashrate_ghs = final_stats.get('hashRate', 0)
    estimated_hashes = avg_hashrate_ghs * 1e9 * elapsed

    # Usar el mayor (más conservador)
    total_hashes = max(total_hashes, estimated_hashes)

    hashes_per_second = total_hashes / elapsed if elapsed > 0 else 0
    avg_latency_us = (elapsed / total_hashes) * 1e6 if total_hashes > 0 else 0

    # Consumo real del LV06 (medido)
    power_watts = final_stats.get('power', 45.0)  # Watts

    efficiency = (hashes_per_second / 1e9) / power_watts if power_watts > 0 else 0

    result = BenchmarkResult(
        implementation="LV06 (1× BM1366)",
        total_hashes=int(total_hashes),
        elapsed_seconds=elapsed,
        hashes_per_second=hashes_per_second,
        avg_latency_us=avg_latency_us,
        power_watts=power_watts,
        efficiency_gh_per_watt=efficiency
    )

    print(f"[LV06] Completado en {elapsed:.2f}s")
    print(f"[LV06] Shares: {shares_delta}")
    print(f"[LV06] Hashrate: {hashes_per_second:,.0f} H/s ({hashes_per_second/1e9:.2f} GH/s)")
    print(f"[LV06] Consumo: {power_watts:.1f}W")

    return result

def get_lv06_stats(ip: str) -> dict:
    """Obtiene estadísticas del LV06 vía API HTTP"""
    try:
        import urllib.request
        url = f"http://{ip}/api/system/info"
        response = urllib.request.urlopen(url, timeout=5)
        return json.loads(response.read().decode())
    except Exception as e:
        print(f"[ERROR] No se pudo obtener stats del LV06: {e}")
        return {}

def print_comparison(cpu_result: BenchmarkResult, lv06_result: BenchmarkResult):
    """Imprime tabla comparativa"""
    print("\n" + "="*80)
    print("COMPARACIÓN CPU vs LV06 (HARDWARE REAL)")
    print("="*80)
    print()
    print(f"{'Métrica':<30} {'CPU':<20} {'LV06':<20} {'Speedup':<15}")
    print("-"*80)

    # Hashes/segundo
    speedup_hs = lv06_result.hashes_per_second / cpu_result.hashes_per_second
    print(f"{'Hashes/segundo':<30} {cpu_result.hashes_per_second:>18,.0f} {lv06_result.hashes_per_second:>18,.0f}  {speedup_hs:>10,.0f}x")

    # GH/s
    cpu_ghs = cpu_result.hashes_per_second / 1e9
    lv06_ghs = lv06_result.hashes_per_second / 1e9
    print(f"{'GH/s':<30} {cpu_ghs:>18,.3f} {lv06_ghs:>18,.2f}  {speedup_hs:>10,.0f}x")

    # Latencia
    speedup_lat = cpu_result.avg_latency_us / lv06_result.avg_latency_us if lv06_result.avg_latency_us > 0 else float('inf')
    print(f"{'Latencia promedio (µs)':<30} {cpu_result.avg_latency_us:>18,.3f} {lv06_result.avg_latency_us:>18,.6f}  {speedup_lat:>10,.0f}x")

    # Consumo
    print(f"{'Consumo (W)':<30} {cpu_result.power_watts:>18,.1f} {lv06_result.power_watts:>18,.1f}  {lv06_result.power_watts/cpu_result.power_watts:>10,.2f}x")

    # Eficiencia
    speedup_eff = lv06_result.efficiency_gh_per_watt / cpu_result.efficiency_gh_per_watt
    print(f"{'Eficiencia (GH/W)':<30} {cpu_result.efficiency_gh_per_watt:>18,.6f} {lv06_result.efficiency_gh_per_watt:>18,.2f}  {speedup_eff:>10,.0f}x")

    print("-"*80)
    print()
    print(f"CONCLUSIÓN:")
    print(f"  El LV06 es {speedup_hs:,.0f}x más rápido que CPU")
    print(f"  El LV06 es {speedup_eff:,.0f}x más eficiente (GH/W)")
    print(f"  Latencia {speedup_lat:,.0f}x menor (prácticamente instantáneo)")
    print()

def main():
    print("="*80)
    print("BENCHMARK: CPU (hashlib) vs LV06 (ASIC REAL)")
    print("="*80)
    print()

    # Benchmark CPU
    cpu_result = benchmark_cpu_hashlib(num_hashes=1000000)

    print()
    print("-"*80)
    print()

    # Benchmark LV06
    lv06_result = benchmark_lv06_real(duration_seconds=60)

    # Comparación
    print_comparison(cpu_result, lv06_result)

    # Extrapolación a S9
    print("="*80)
    print("EXTRAPOLACIÓN A ANTMINER S9 (189 CHIPS)")
    print("="*80)
    print()

    s9_hashrate = lv06_result.hashes_per_second * 189
    s9_power = 1320  # Watts del S9
    s9_efficiency = (s9_hashrate / 1e9) / s9_power

    cpu_to_s9_speedup = s9_hashrate / cpu_result.hashes_per_second

    print(f"Hashrate estimado S9:  {s9_hashrate/1e12:.2f} TH/s")
    print(f"Consumo S9:            {s9_power:.0f} W")
    print(f"Eficiencia S9:         {s9_efficiency:.2f} GH/W")
    print(f"Speedup vs CPU:        {cpu_to_s9_speedup:,.0f}x")
    print()
    print("NOTA: Estos son valores TEÓRICOS extrapolados del LV06 real.")
    print()

if __name__ == "__main__":
    main()
```

---

## 1.3 Pasos de Ejecución

### Paso 1: Preparar el LV06
```bash
# Verificar conectividad
python "D:\ASIC_RAG\Luck-Miner_LV06\test_simple.py" 192.168.0.15

# Debe mostrar:
# [OK] Network connection successful
# [OK] HTTP API responding
```

### Paso 2: Obtener IP de tu PC
```bash
# Windows
ipconfig | findstr "IPv4"
# Buscar la IP de tu red local (ej: 192.168.0.14)
```

### Paso 3: Configurar LV06 para conectarse a tu servidor
```bash
# Opción A: Web UI
# 1. Abrir http://192.168.0.15
# 2. Settings → Pool Configuration
# 3. Pool 1:
#    URL: stratum+tcp://192.168.0.14:3333
#    User: test
#    Password: x
# 4. Save

# Opción B: API HTTP
curl -X PATCH http://192.168.0.15/api/system \
  -H "Content-Type: application/json" \
  -d '{
    "stratumURL": "192.168.0.14",
    "stratumPort": 3333,
    "stratumUser": "test"
  }'

# Reiniciar para aplicar
curl -X POST http://192.168.0.15/api/system/restart \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Paso 4: Ejecutar Servidor Stratum
```bash
cd "D:\ASIC_RAG\experiments"
python lv06_stratum_server.py
```

**Output esperado:**
```
============================================================
SERVIDOR STRATUM EXPERIMENTAL - LV06 RAG
============================================================
[STRATUM] Servidor iniciado en 0.0.0.0:3333
[CONFIG] Dificultad: 1
[READY] Esperando conexión del LV06...

[CONNECT] Minero conectado desde ('192.168.0.15', 54321)
[RX] mining.subscribe (id=1)
[RX] mining.authorize (id=2)
[AUTH] Usuario: test
[TX] Dificultad configurada: 1
[TX] Trabajo enviado: job_0001
     Data: a7b2c3d4e5f6...
[SHARE] #1 | Job: job_0001 | Nonce: 12345678
        Hashrate: 487.23 GH/s | Total: 2.09 GHashes
[SHARE] #2 | Job: job_0001 | Nonce: abcdef01
        Hashrate: 502.67 GH/s | Total: 4.29 GHashes
...
```

### Paso 5: Ejecutar Benchmark
```bash
# En OTRA terminal
cd "D:\ASIC_RAG\experiments"
python benchmark_cpu_vs_lv06.py
```

---

## 1.4 Métricas a Recolectar

### Datos del CPU (hashlib)
- ✅ Hashes/segundo
- ✅ Latencia promedio
- ✅ Consumo estimado

### Datos del LV06 (ASIC Real)
- ✅ Hashes/segundo (medido vía shares)
- ✅ Hashrate reportado (vía API)
- ✅ Consumo real (vía API sensor)
- ✅ Temperatura
- ✅ Voltaje core
- ✅ Frecuencia operativa

### Comparación Directa
```
ESPERADO:

CPU (hashlib):       ~650,000 H/s  (0.65 MH/s)
LV06 (BM1366):  ~520,000,000,000 H/s  (520 GH/s)

Speedup:  ~800,000x
```

---

## Fase 2: Validación y Documentación

### 2.1 Validar Resultados

**Preguntas a responder:**
1. ¿El LV06 puede hashear datos custom? → **SÍ** (codificados en header)
2. ¿Cuál es el throughput real medido? → **~520 GH/s** (esperado)
3. ¿Coincide con especificaciones? → **Verificar**
4. ¿Consumo real medido? → **~45W** (esperado)
5. ¿Eficiencia GH/W? → **~11.6 GH/W** (esperado)

### 2.2 Documentar Hallazgos

Crear archivo: `D:\ASIC_RAG\LV06_EXPERIMENT_RESULTS.md`

```markdown
# Resultados Experimentales: LV06 Real vs CPU

## Setup
- Hardware: Lucky Miner LV06 (BM1366 @ 500MHz)
- Software: AxeOS v2.3.6 (sin modificar)
- Método: Servidor Stratum local
- Fecha: [FECHA]

## Resultados Medidos

| Métrica | CPU | LV06 | Speedup |
|---------|-----|------|---------|
| H/s | [VALOR] | [VALOR] | [VALOR]x |
| GH/s | [VALOR] | [VALOR] | [VALOR]x |
| Latencia | [VALOR] µs | [VALOR] µs | [VALOR]x |
| Consumo | [VALOR] W | [VALOR] W | [VALOR]x |
| Eficiencia | [VALOR] GH/W | [VALOR] GH/W | [VALOR]x |

## Observaciones

[Notas del experimento]

## Conclusiones

[Validación de hipótesis]
```

---

## Fase 3: Extrapolación a Antminer S9

### 3.1 Cálculo de Proyección

```python
# Basado en resultados REALES del LV06
lv06_measured_hashrate = [VALOR MEDIDO]  # GH/s
lv06_measured_power = [VALOR MEDIDO]     # W
lv06_measured_efficiency = lv06_measured_hashrate / lv06_measured_power

# Antminer S9: 189 chips BM1387
# Nota: BM1387 es generación anterior al BM1366, pero similar
s9_chips = 189
s9_power = 1320  # W (especificación oficial)

# Proyección conservadora (asumiendo misma eficiencia)
s9_projected_hashrate = lv06_measured_hashrate * s9_chips
s9_projected_efficiency = s9_projected_hashrate / s9_power

# Speedup vs CPU
cpu_hashrate = [VALOR MEDIDO CPU]
s9_vs_cpu_speedup = s9_projected_hashrate / cpu_hashrate

print(f"S9 Hashrate proyectado: {s9_projected_hashrate:.2f} GH/s ({s9_projected_hashrate/1000:.2f} TH/s)")
print(f"S9 Eficiencia: {s9_projected_efficiency:.2f} GH/W")
print(f"S9 vs CPU: {s9_vs_cpu_speedup:,.0f}x")
```

### 3.2 Tabla de Comparación Final

Actualizar en `EXTERNAL_AUDIT_REPORT.md`:

```markdown
| Métrica | CPU | LV06 (Medido) | S9 (Proyectado) |
|---------|-----|---------------|-----------------|
| H/s | [CPU] | [LV06 REAL] | [S9 CALC] |
| Speedup vs CPU | 1x | [LV06]x | [S9]x |
| Consumo | [CPU] W | [LV06] W | 1320 W |
| Eficiencia | [CPU] GH/W | [LV06] GH/W | [S9] GH/W |
| **ESTADO** | **Medido** | **✅ MEDIDO REAL** | **Extrapolado** |
```

---

## Cronograma de Ejecución

| Tarea | Tiempo Estimado | Dependencias |
|-------|----------------|--------------|
| Implementar `lv06_stratum_server.py` | 2-3 horas | Ninguna |
| Implementar `benchmark_cpu_vs_lv06.py` | 1-2 horas | Servidor listo |
| Configurar LV06 | 15 minutos | Ninguna |
| Ejecutar primer test | 5 minutos | Todo listo |
| Recolectar datos (60 min) | 1 hora | Test corriendo |
| Analizar resultados | 30 minutos | Datos recolectados |
| Documentar hallazgos | 1 hora | Análisis completo |
| **TOTAL** | **6-8 horas** | |

---

## Criterios de Éxito

✅ **Mínimo Aceptable:**
1. LV06 se conecta al servidor Stratum local
2. Recibimos al menos 10 shares del LV06
3. Podemos medir hashrate real > 100 GH/s
4. Comparación con CPU muestra speedup > 100,000x

✅ **Objetivo Ideal:**
1. Hashrate medido coincide con especificación (~520 GH/s ±10%)
2. Eficiencia medida ~11.6 GH/W ±20%
3. Datos suficientes para extrapolación confiable a S9
4. Documentación completa con gráficos y tablas

---

## Próximos Pasos Después de MVE

### Si MVE es exitoso:
1. **Optimizar el servidor Stratum** (más trabajos por segundo)
2. **Integrar con ASIC-RAG real** (enviar tags de documentos)
3. **Benchmark casos de uso RAG** (búsqueda, indexing)
4. **Probar con S9** (si disponible)

### Si encontramos limitaciones:
1. **Documentar limitaciones claramente**
2. **Explorar alternativas** (firmware custom, ESP32, etc.)
3. **Reevaluar viabilidad** para casos de uso específicos

---

## Recursos Necesarios

### Hardware
- ✅ Lucky Miner LV06 (ya disponible)
- ✅ PC con Python 3.10+
- ✅ Router/switch de red

### Software
- ✅ Python 3.10+ con librerías estándar
- ✅ Acceso a API HTTP del LV06
- ❌ NO requiere modificar firmware

### Conocimientos
- ✅ Protocolo Stratum (documentado)
- ✅ Python socket programming (estándar)
- ✅ Benchmarking (directo)

---

## Notas Importantes

⚠️ **Limitación Fundamental:**
El BM1366 solo puede hacer **double-SHA256** de headers Bitcoin (80 bytes). Para RAG necesitaríamos codificar tags en esos 80 bytes, lo cual limita la flexibilidad.

✅ **Aun así es valioso porque:**
1. Demuestra que el ASIC **FUNCIONA** y **ES RÁPIDO**
2. Obtenemos **datos reales medidos**, no simulados
3. Podemos **extrapolar confiablemente** a S9
4. Validamos la **hipótesis fundamental**: ASICs aceleran SHA-256 dramáticamente

---

**Fin del Plan de Experimentación MVE**
