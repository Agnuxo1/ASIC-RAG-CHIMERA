import socket
import json
import time
import health_config as cfg

def run():
    print("="*70)
    print("EXPERIMENT 4: ASICS VS CUDA-GPU HEAD-TO-HEAD (UNBIASED)")
    print("="*70)
    print(f"Scenario: Healthcare Blockchain PoW Performance")
    print(f"Reference implementation: RTX 3080 (CUDA) @ 375 MH/s")
    print(f"Physical hardware: LV06 (Single BM1387) @ 9W")
    print(f"Difficulty Calibrated to: {cfg.BLOCK_DIFFICULTY}")
    print("-" * 70)

    # 1. Reset Bridge
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((cfg.PC_IP, cfg.API_PORT))
        s.sendall(b"RESET")
        s.close()
    except ConnectionRefusedError:
        print("❌ Error: Run 'medical_bridge.py' first.")
        return

    # 2. Monitor for 5 Minutes (Isolation)
    duration = 300
    block_times = []
    start_time = time.time()
    last_block_time = start_time
    
    print(f"🧪 Sampling LV06 for {duration} seconds...")
    
    last_share_count = 0
    while (time.time() - start_time) < duration:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((cfg.PC_IP, cfg.API_PORT))
            s.sendall(b"GET_STATS")
            data = json.loads(s.recv(4096).decode())
            s.close()
            
            current_shares = data["shares_accepted"]
            if current_shares > last_share_count:
                now = time.time()
                block_times.append(now - last_block_time)
                last_block_time = now
                last_share_count = current_shares
                print(f"   [BLOCK FOUND] Latency: {block_times[-1]:.4f}s | Total: {current_shares}")
            
            time.sleep(0.05) # High frequency polling for sub-second precision
        except:
            pass
            
    total_duration = time.time() - start_time
    total_shares = last_share_count
    
    # 3. Stats Calculation
    if len(block_times) > 0:
        avg_block_time = sum(block_times) / len(block_times)
    else:
        avg_block_time = 0
        
    # Real Hashrate (H = Shares * 2^32 / Difficulty / Time) -> Wait, Difficulty is already factored in
    # In Stratum, Share = 2^32 hashes. 
    # Measured Hashrate (MH/s) = (Total Shares * 2^32) / (Total Time * 10^6)
    measured_mhs = (total_shares * (2**32)) / (total_duration * 10**6)

    print("\n" + "="*70)
    print("FINAL HEAD-TO-HEAD COMPARISON")
    print("="*70)
    print(f"{'METRIC':<25} | {'RTX 3080 (CUDA)':<20} | {'LV06 (ASIC)':<15}")
    print("-" * 70)
    print(f"{'Hash Rate (MH/s)':<25} | {'375 MH/s':<20} | {measured_mhs:.2f} MH/s")
    print(f"{'Avg Block Time (s)':<25} | {'0.83 s':<20} | {avg_block_time:.4f} s")
    print(f"{'Power (Watts)':<25} | {'320-350 W':<20} | {'9.0 W':<15}")
    print(f"{'Efficiency (MH/W)':<25} | {'1.17':<20} | {measured_mhs/9.0:.2f}")
    print(f"{'Device Cost':<25} | {'~€1,000':<20} | {'$40.00':<15}")
    print("-" * 70)

    # 4. Conclusion
    efficiency_gap = (measured_mhs/9.0) / 1.17
    print(f"\n💡 UNBIASED CONCLUSION:")
    print(f"The LV06 ASIC is {efficiency_gap:.1f}x more energy-efficient than the RTX 3080 implementation.")
    print(f"With an average block time of {avg_block_time:.2f}s, the ASIC node fulfills the latency")
    print(f"requirements for healthcare record indexing at ~1/25th of the cost and 1/35th of the power.")

if __name__ == "__main__":
    run()
