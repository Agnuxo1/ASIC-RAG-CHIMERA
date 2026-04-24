#!/usr/bin/env python3
"""
================================================================================
LV06 CALIBRATION BENCHMARK
================================================================================
Encuentra automáticamente los parámetros óptimos para el LV06:
  1. Rango de dificultad aceptado por el chip
  2. D_BASE óptimo para máximo share rate
  3. Respuesta diferencial (¿el chip responde a cambios de D?)
  4. WINDOW_TIME óptimo para el cuello de botella WiFi

Salida:
  - calibration/cal_YYYYMMDD_HHMMSS.json (resultados completos)
  - calibration/cal_YYYYMMDD_HHMMSS.log (log completo)
  - Recomendaciones de parámetros para experimento principal

Author: Fran / Agnuxo + Claude
Date: December 2025
================================================================================
"""

import socket
import threading
import json
import time
import struct
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================
class CalibrationConfig:
    # Network
    HOST = "0.0.0.0"
    PORT = 3333
    HANDSHAKE_TIMEOUT = 300.0
    
    # Difficulty sweep range (logarithmic)
    # Probaremos: 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    DIFFICULTY_TEST_VALUES = [
        0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
        1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0
    ]
    
    # Time per difficulty test
    SAMPLES_PER_DIFFICULTY = 10  # Repeticiones por dificultad
    WINDOW_TIME_TEST = 3.0       # Segundos por muestra
    
    # Window time sweep
    WINDOW_TIMES_TO_TEST = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Response test
    RESPONSE_TEST_CYCLES = 20    # Ciclos bajo/alto para test de respuesta

# =============================================================================
# LOGGER
# =============================================================================
class CalibrationLogger:
    def __init__(self):
        self.cal_dir = "calibration"
        os.makedirs(self.cal_dir, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(self.cal_dir, f"cal_{self.run_id}.json")
        self.log_path = os.path.join(self.cal_dir, f"cal_{self.run_id}.log")
        
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.log(f"=== CALIBRATION RUN {self.run_id} ===")
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        self.log_file.write(formatted + "\n")
        self.log_file.flush()
        
    def save(self, data: Dict):
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.log(f"[SAVED] {self.json_path}")
        
    def close(self):
        self.log_file.close()

# =============================================================================
# STRATUM SERVER (Minimal for calibration)
# =============================================================================
class CalibrationServer(threading.Thread):
    def __init__(self, config: CalibrationConfig, logger: CalibrationLogger):
        super().__init__(daemon=True)
        self.config = config
        self.logger = logger
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((config.HOST, config.PORT))
        self.sock.listen(1)
        
        self.client_conn = None
        self.running = True
        self.authorized = False
        self.share_buffer = []
        self.buffer_lock = threading.Lock()
        self.job_counter = 0
        self.current_difficulty = 1.0
        
        # Stats
        self.shares_accepted = 0
        self.shares_rejected = 0
        
    def run(self):
        self.logger.log(f"[SERVER] Listening on port {self.config.PORT}")
        try:
            self.sock.settimeout(self.config.HANDSHAKE_TIMEOUT)
            conn, addr = self.sock.accept()
            self.logger.log(f"[SERVER] Connected: {addr}")
            self.client_conn = conn
            self._handle_client(conn)
        except socket.timeout:
            self.logger.log("[SERVER] Timeout waiting for miner")
        except Exception as e:
            self.logger.log(f"[SERVER] Error: {e}")

    def _handle_client(self, conn):
        conn.settimeout(0.1)
        buffer = ""
        while self.running:
            try:
                data = conn.recv(8192)
                if not data: break
                buffer += data.decode('utf-8', errors='ignore')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process(conn, line.strip())
            except socket.timeout:
                continue
            except: break

    def _process(self, conn, line):
        if not line: return
        try: msg = json.loads(line)
        except: return
        
        method, mid = msg.get('method', ''), msg.get('id')
        
        if method == 'mining.subscribe':
            self._send(conn, {
                "id": mid, 
                "result": [[["mining.set_difficulty","1"],["mining.notify","1"]], "08000002", 4], 
                "error": None
            })
            # Enviar dificultad inicial
            self._send(conn, {"id": None, "method": "mining.set_difficulty", "params": [self.current_difficulty]})
            
        elif method == 'mining.authorize':
            self._send(conn, {"id": mid, "result": True, "error": None})
            self.authorized = True
            self.logger.log("[SERVER] Miner authorized")
            
        elif method == 'mining.configure':
            self._send(conn, {"id": mid, "result": {}, "error": None})
            
        elif method == 'mining.submit':
            arrival = time.perf_counter()
            with self.buffer_lock:
                self.share_buffer.append({
                    "time": arrival,
                    "difficulty": self.current_difficulty
                })
            self.shares_accepted += 1
            self._send(conn, {"id": mid, "result": True, "error": None})

    def _send(self, conn, data):
        try: 
            conn.sendall((json.dumps(data) + '\n').encode())
        except: pass

    def set_difficulty(self, diff: float):
        """Cambiar dificultad y enviar nuevo job"""
        if not self.client_conn or not self.authorized:
            return False
            
        self.current_difficulty = diff
        
        # Enviar nueva dificultad
        self._send(self.client_conn, {
            "id": None, 
            "method": "mining.set_difficulty", 
            "params": [diff]
        })
        
        # Pequeña pausa para que procese
        time.sleep(0.02)
        
        # Enviar nuevo job para forzar el cambio
        self.job_counter += 1
        job_id = str(self.job_counter)
        
        # Coinbase simple
        coinb1 = "0100000001" + "00"*32 + "ffffffff10" + "04" + "deadbeef" + "0a" + "00"*10
        coinb2 = "ffffffff01" + "00f2052a01000000" + "00"*8
        ntime = hex(int(time.time()))[2:].zfill(8)
        
        self._send(self.client_conn, {
            "id": None, 
            "method": "mining.notify",
            "params": [job_id, "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
        })
        
        return True

    def harvest(self, window: float) -> List[Dict]:
        """Recoger shares durante window segundos"""
        time.sleep(window)
        with self.buffer_lock:
            shares = list(self.share_buffer)
            self.share_buffer = []
        return shares

    def clear_buffer(self):
        with self.buffer_lock:
            self.share_buffer = []

    def stop(self):
        self.running = False
        try: self.sock.close()
        except: pass

# =============================================================================
# CALIBRATION TESTS
# =============================================================================
def run_difficulty_sweep(server: CalibrationServer, config: CalibrationConfig, logger: CalibrationLogger) -> Dict:
    """
    TEST 1: Barrido de dificultades
    Encuentra qué dificultades producen shares
    """
    logger.log("\n" + "="*60)
    logger.log("   TEST 1: DIFFICULTY SWEEP")
    logger.log("   Finding accepted difficulty range")
    logger.log("="*60)
    
    results = {}
    
    for diff in config.DIFFICULTY_TEST_VALUES:
        logger.log(f"\n[TEST] Difficulty = {diff}")
        
        share_counts = []
        
        for sample in range(config.SAMPLES_PER_DIFFICULTY):
            server.clear_buffer()
            server.set_difficulty(diff)
            shares = server.harvest(config.WINDOW_TIME_TEST)
            share_counts.append(len(shares))
            
            if sample == 0 or (sample + 1) % 5 == 0:
                logger.log(f"  Sample {sample+1}/{config.SAMPLES_PER_DIFFICULTY}: {len(shares)} shares")
        
        avg_shares = sum(share_counts) / len(share_counts)
        total_shares = sum(share_counts)
        
        results[diff] = {
            "samples": share_counts,
            "avg_shares_per_window": avg_shares,
            "total_shares": total_shares,
            "share_rate": avg_shares / config.WINDOW_TIME_TEST
        }
        
        logger.log(f"  → Avg: {avg_shares:.2f} shares/window ({avg_shares/config.WINDOW_TIME_TEST:.3f} shares/sec)")
    
    return results

def find_valid_range(sweep_results: Dict, logger: CalibrationLogger) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Analiza el barrido y encuentra:
    - D_MIN: Dificultad mínima que produce shares
    - D_MAX: Dificultad máxima que produce shares
    - D_OPTIMAL: Dificultad con mejor share rate
    """
    valid_diffs = []
    
    for diff, data in sweep_results.items():
        if data['total_shares'] > 0:
            valid_diffs.append((diff, data['share_rate']))
    
    if not valid_diffs:
        logger.log("\n[ERROR] No valid difficulties found! Chip not responding.")
        return None, None, None
    
    valid_diffs.sort(key=lambda x: x[0])  # Sort by difficulty
    
    d_min = valid_diffs[0][0]
    d_max = valid_diffs[-1][0]
    d_optimal = max(valid_diffs, key=lambda x: x[1])[0]  # Best share rate
    
    logger.log(f"\n[ANALYSIS] Valid difficulty range:")
    logger.log(f"  D_MIN (lowest working): {d_min}")
    logger.log(f"  D_MAX (highest working): {d_max}")
    logger.log(f"  D_OPTIMAL (best rate): {d_optimal}")
    
    return d_min, d_max, d_optimal

def run_response_test(server: CalibrationServer, config: CalibrationConfig, logger: CalibrationLogger, 
                      d_low: float, d_high: float) -> Dict:
    """
    TEST 2: Test de respuesta diferencial
    ¿El chip responde diferente a D_LOW vs D_HIGH?
    """
    logger.log("\n" + "="*60)
    logger.log("   TEST 2: DIFFERENTIAL RESPONSE TEST")
    logger.log(f"   Alternating D_LOW={d_low} vs D_HIGH={d_high}")
    logger.log("="*60)
    
    low_shares = []
    high_shares = []
    
    for cycle in range(config.RESPONSE_TEST_CYCLES):
        # Low difficulty
        server.clear_buffer()
        server.set_difficulty(d_low)
        shares_low = server.harvest(config.WINDOW_TIME_TEST)
        low_shares.append(len(shares_low))
        
        # High difficulty
        server.clear_buffer()
        server.set_difficulty(d_high)
        shares_high = server.harvest(config.WINDOW_TIME_TEST)
        high_shares.append(len(shares_high))
        
        if (cycle + 1) % 5 == 0:
            logger.log(f"  Cycle {cycle+1}/{config.RESPONSE_TEST_CYCLES}: LOW={len(shares_low)}, HIGH={len(shares_high)}")
    
    avg_low = sum(low_shares) / len(low_shares)
    avg_high = sum(high_shares) / len(high_shares)
    
    # Statistical difference
    if avg_high > 0:
        ratio = avg_low / avg_high
    else:
        ratio = float('inf') if avg_low > 0 else 1.0
    
    # ¿Hay respuesta diferencial?
    responsive = abs(avg_low - avg_high) > 0.5  # Al menos 0.5 shares de diferencia
    
    results = {
        "d_low": d_low,
        "d_high": d_high,
        "low_shares": low_shares,
        "high_shares": high_shares,
        "avg_low": avg_low,
        "avg_high": avg_high,
        "ratio": ratio,
        "responsive": responsive
    }
    
    logger.log(f"\n[RESULT]")
    logger.log(f"  D_LOW ({d_low}): avg {avg_low:.2f} shares/window")
    logger.log(f"  D_HIGH ({d_high}): avg {avg_high:.2f} shares/window")
    logger.log(f"  Ratio: {ratio:.2f}x")
    logger.log(f"  Chip responsive: {'✅ YES' if responsive else '❌ NO'}")
    
    return results

def run_window_sweep(server: CalibrationServer, config: CalibrationConfig, logger: CalibrationLogger,
                     d_optimal: float) -> Dict:
    """
    TEST 3: Barrido de ventanas temporales
    Encuentra el WINDOW_TIME óptimo para capturar shares sin perderlos
    """
    logger.log("\n" + "="*60)
    logger.log("   TEST 3: WINDOW TIME SWEEP")
    logger.log(f"   Testing at D={d_optimal}")
    logger.log("="*60)
    
    results = {}
    
    for window in config.WINDOW_TIMES_TO_TEST:
        logger.log(f"\n[TEST] Window = {window}s")
        
        server.set_difficulty(d_optimal)
        share_counts = []
        
        for sample in range(5):  # 5 samples per window
            server.clear_buffer()
            shares = server.harvest(window)
            share_counts.append(len(shares))
        
        avg_shares = sum(share_counts) / len(share_counts)
        rate = avg_shares / window
        
        results[window] = {
            "samples": share_counts,
            "avg_shares": avg_shares,
            "share_rate": rate
        }
        
        logger.log(f"  → {avg_shares:.2f} shares/window ({rate:.3f} shares/sec)")
    
    return results

def generate_recommendations(sweep_results: Dict, response_results: Dict, 
                            window_results: Dict, logger: CalibrationLogger) -> Dict:
    """
    Genera recomendaciones de parámetros basadas en los tests
    """
    logger.log("\n" + "="*60)
    logger.log("   RECOMMENDATIONS")
    logger.log("="*60)
    
    # Encontrar mejor configuración
    d_min, d_max, d_optimal = find_valid_range(sweep_results, logger)
    
    if d_min is None:
        logger.log("\n❌ CALIBRATION FAILED: No valid difficulty range found")
        return {"status": "FAILED", "reason": "No valid difficulty range"}
    
    # Mejor window
    best_window = max(window_results.items(), key=lambda x: x[1]['share_rate'])[0]
    
    # ¿El chip responde?
    chip_responsive = response_results.get('responsive', False)
    
    recommendations = {
        "status": "SUCCESS" if chip_responsive else "PARTIAL",
        "difficulty": {
            "D_MIN": d_min,
            "D_MAX": d_max,
            "D_OPTIMAL": d_optimal,
            "recommended_D_BASE": d_optimal
        },
        "timing": {
            "best_window": best_window,
            "recommended_WINDOW_TIME": best_window
        },
        "chip_responsive": chip_responsive,
        "experiment_config": {
            "D_BASE": d_optimal,
            "WINDOW_TIME": best_window,
            "EPSILON": d_optimal * 0.05,  # 5% of D_BASE
            "expected_shares_per_step": window_results[best_window]['avg_shares']
        }
    }
    
    logger.log(f"\n📋 RECOMMENDED PARAMETERS FOR EXPERIMENT:")
    logger.log(f"   D_BASE = {d_optimal}")
    logger.log(f"   WINDOW_TIME = {best_window}s")
    logger.log(f"   EPSILON = {d_optimal * 0.05}")
    logger.log(f"   Expected shares/step: ~{window_results[best_window]['avg_shares']:.1f}")
    
    if chip_responsive:
        logger.log(f"\n✅ Chip shows differential response to difficulty changes")
        logger.log(f"   The experiment should work with these parameters")
    else:
        logger.log(f"\n⚠️ Chip does NOT show clear differential response")
        logger.log(f"   Experiment may not work - check hardware settings")
    
    return recommendations

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = CalibrationLogger()
    config = CalibrationConfig()
    
    logger.log("="*60)
    logger.log("   LV06 CALIBRATION BENCHMARK")
    logger.log("   Finding optimal parameters for reservoir experiment")
    logger.log("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "difficulties_tested": config.DIFFICULTY_TEST_VALUES,
            "samples_per_difficulty": config.SAMPLES_PER_DIFFICULTY,
            "window_time_test": config.WINDOW_TIME_TEST
        }
    }
    
    server = CalibrationServer(config, logger)
    server.start()
    
    # Wait for miner
    logger.log("\n[WAIT] Waiting for miner connection...")
    logger.log("       Configure LV06 to connect to this IP on port 3333")
    
    timeout_start = time.time()
    while not server.authorized:
        if time.time() - timeout_start > config.HANDSHAKE_TIMEOUT:
            logger.log("[ERROR] Timeout waiting for miner")
            results["status"] = "TIMEOUT"
            logger.save(results)
            return
        time.sleep(0.5)
    
    # Warmup
    logger.log("\n[WARMUP] 10 seconds warmup...")
    server.set_difficulty(1.0)
    server.harvest(10.0)
    
    try:
        # TEST 1: Difficulty sweep
        sweep_results = run_difficulty_sweep(server, config, logger)
        results["difficulty_sweep"] = sweep_results
        
        # Find valid range
        d_min, d_max, d_optimal = find_valid_range(sweep_results, logger)
        
        if d_optimal is None:
            logger.log("\n[ABORT] Cannot continue without valid difficulty range")
            results["status"] = "NO_VALID_RANGE"
            logger.save(results)
            return
        
        # TEST 2: Response test
        # Use d_optimal as low and 10x as high (within valid range)
        d_test_high = min(d_optimal * 10, d_max) if d_max else d_optimal * 10
        response_results = run_response_test(server, config, logger, d_optimal, d_test_high)
        results["response_test"] = response_results
        
        # TEST 3: Window sweep
        window_results = run_window_sweep(server, config, logger, d_optimal)
        results["window_sweep"] = window_results
        
        # Generate recommendations
        recommendations = generate_recommendations(sweep_results, response_results, window_results, logger)
        results["recommendations"] = recommendations
        
        # Final summary
        logger.log("\n" + "="*60)
        logger.log("   CALIBRATION COMPLETE")
        logger.log("="*60)
        
        if recommendations["status"] == "SUCCESS":
            logger.log("\n✅ LV06 is ready for reservoir experiment")
            logger.log(f"\nUse these parameters in lv06_experiment_v6:")
            logger.log(f"   D_BASE = {recommendations['experiment_config']['D_BASE']}")
            logger.log(f"   WINDOW_TIME = {recommendations['experiment_config']['WINDOW_TIME']}")
            logger.log(f"   EPSILON = {recommendations['experiment_config']['EPSILON']}")
        else:
            logger.log("\n⚠️ Calibration completed with warnings")
            logger.log("   Review results before running experiment")
        
    except KeyboardInterrupt:
        logger.log("\n[INTERRUPTED] Calibration stopped by user")
        results["status"] = "INTERRUPTED"
    except Exception as e:
        logger.log(f"\n[ERROR] {e}")
        results["status"] = f"ERROR: {e}"
    finally:
        server.stop()
    
    logger.save(results)
    logger.close()
    
    print(f"\n📁 Results saved to: {logger.json_path}")
    print(f"📁 Log saved to: {logger.log_path}")

if __name__ == "__main__":
    main()
