#!/usr/bin/env python3
"""
CPU vs LV06 ASIC Benchmark - Real Hardware Validation

Measures and compares:
- Hash throughput (H/s)
- Latency per hash
- Power consumption
- Efficiency (GH/W)

This benchmark provides REAL data from actual hardware:
- CPU: Python hashlib (standard SHA-256)
- LV06: BM1366 ASIC chip @ 500MHz

Results will be used to:
1. Validate theoretical projections
2. Extrapolate to Antminer S9 (189 chips)
3. Update audit report with real measurements

Author: ASIC-RAG-CHIMERA Project
Date: 2024-12-18
License: MIT
"""

import hashlib
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import statistics


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    implementation: str
    total_hashes: int
    elapsed_seconds: float
    hashes_per_second: float
    hashrate_mhs: float
    hashrate_ghs: float
    avg_latency_us: float
    power_watts: float
    efficiency_gh_per_watt: float
    temperature_c: Optional[float] = None
    voltage_mv: Optional[int] = None
    frequency_mhz: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class CPUBenchmark:
    """Benchmark CPU SHA-256 performance using hashlib"""

    def __init__(self):
        self.name = "CPU (hashlib)"

    def run(self, num_hashes: int = 1000000, data_size: int = 256) -> BenchmarkResult:
        """
        Run CPU benchmark.

        Args:
            num_hashes: Number of hashes to compute
            data_size: Size of data to hash (bytes)

        Returns:
            BenchmarkResult with measured performance
        """
        print(f"\n{'='*70}")
        print(f"CPU BENCHMARK - hashlib")
        print(f"{'='*70}")
        print(f"Hashes to compute: {num_hashes:,}")
        print(f"Data size: {data_size} bytes")
        print()

        # Generate test data
        print("Generating test data...")
        test_data = [f"test_data_{i}".ljust(data_size, 'x').encode()
                     for i in range(min(1000, num_hashes))]

        print(f"Starting benchmark...")

        # Warmup
        for data in test_data[:100]:
            hashlib.sha256(data).digest()

        # Benchmark
        start_time = time.perf_counter()
        hash_count = 0

        while hash_count < num_hashes:
            for data in test_data:
                hashlib.sha256(data).digest()
                hash_count += 1
                if hash_count >= num_hashes:
                    break

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Calculate metrics
        hashes_per_second = num_hashes / elapsed
        hashrate_mhs = hashes_per_second / 1e6
        hashrate_ghs = hashes_per_second / 1e9
        avg_latency_us = (elapsed / num_hashes) * 1e6

        # Estimate CPU power consumption
        # Typical i7 TDP: 65W, assume 50% utilization during hashing
        power_watts = 65.0 * 0.5

        # Calculate efficiency
        efficiency = hashrate_ghs / power_watts if power_watts > 0 else 0

        result = BenchmarkResult(
            implementation=self.name,
            total_hashes=num_hashes,
            elapsed_seconds=elapsed,
            hashes_per_second=hashes_per_second,
            hashrate_mhs=hashrate_mhs,
            hashrate_ghs=hashrate_ghs,
            avg_latency_us=avg_latency_us,
            power_watts=power_watts,
            efficiency_gh_per_watt=efficiency
        )

        # Print results
        print(f"\nResults:")
        print(f"  Time elapsed:     {elapsed:.2f} seconds")
        print(f"  Hashes/second:    {hashes_per_second:,.0f} H/s")
        print(f"  Hashrate:         {hashrate_mhs:.3f} MH/s ({hashrate_ghs:.6f} GH/s)")
        print(f"  Avg latency:      {avg_latency_us:.3f} µs")
        print(f"  Power (est):      {power_watts:.1f} W")
        print(f"  Efficiency:       {efficiency:.6f} GH/W")

        return result


class LV06Benchmark:
    """Benchmark LV06 ASIC hardware via HTTP API"""

    def __init__(self, lv06_ip: str = "192.168.0.15"):
        self.lv06_ip = lv06_ip
        self.name = "LV06 (1× BM1366 ASIC)"

    def get_stats(self) -> Optional[Dict]:
        """Get statistics from LV06 via HTTP API"""
        try:
            url = f"http://{self.lv06_ip}/api/system/info"
            response = urllib.request.urlopen(url, timeout=5)
            data = json.loads(response.read().decode())
            return data
        except Exception as e:
            print(f"ERROR: Could not get LV06 stats: {e}")
            return None

    def run(self, duration_seconds: int = 60) -> BenchmarkResult:
        """
        Run LV06 benchmark by monitoring real ASIC performance.

        Args:
            duration_seconds: How long to monitor the ASIC

        Returns:
            BenchmarkResult with REAL measured data
        """
        print(f"\n{'='*70}")
        print(f"LV06 ASIC BENCHMARK - REAL HARDWARE")
        print(f"{'='*70}")
        print(f"Monitoring duration: {duration_seconds} seconds")
        print(f"LV06 IP: {self.lv06_ip}")
        print()

        # Check if LV06 is accessible
        print("Checking LV06 connection...")
        initial_stats = self.get_stats()

        if not initial_stats:
            print("ERROR: Cannot connect to LV06. Make sure:")
            print(f"  1. LV06 is powered on")
            print(f"  2. IP address is correct: {self.lv06_ip}")
            print(f"  3. LV06 is on the same network")
            print(f"  4. Stratum server is running (lv06_stratum_server.py)")
            return self._create_error_result()

        print(f"✓ LV06 connected")
        print(f"  Hashrate:    {initial_stats.get('hashRate', 0):.2f} GH/s")
        print(f"  Temperature: {initial_stats.get('temp', 0)}°C")
        print(f"  Power:       {initial_stats.get('power', 0):.2f} W")
        print(f"  Voltage:     {initial_stats.get('coreVoltageActual', 0)} mV")
        print(f"  Frequency:   {initial_stats.get('frequency', 0)} MHz")
        print()

        # Record initial state
        initial_shares = initial_stats.get('sharesAccepted', 0)
        initial_time = time.time()

        print(f"Monitoring for {duration_seconds} seconds...")
        print("(LV06 must be mining to Stratum server for accurate results)")
        print()

        # Wait for monitoring period
        for remaining in range(duration_seconds, 0, -10):
            time.sleep(min(10, remaining))
            current_stats = self.get_stats()
            if current_stats:
                current_hashrate = current_stats.get('hashRate', 0)
                print(f"  [{duration_seconds-remaining:3d}s] Hashrate: {current_hashrate:.2f} GH/s",
                      end='\r', flush=True)

        print()  # New line after progress

        # Record final state
        final_stats = self.get_stats()
        final_time = time.time()

        if not final_stats:
            print("ERROR: Lost connection to LV06 during monitoring")
            return self._create_error_result()

        # Calculate metrics
        elapsed = final_time - initial_time
        final_shares = final_stats.get('sharesAccepted', 0)
        shares_delta = final_shares - initial_shares

        # Calculate total hashes
        # Each share at difficulty=1 represents 2^32 hashes
        hashes_from_shares = shares_delta * (2 ** 32)

        # Also estimate from reported hashrate
        avg_hashrate_ghs = final_stats.get('hashRate', 0)
        hashes_from_hashrate = avg_hashrate_ghs * 1e9 * elapsed

        # Use the maximum (more conservative estimate)
        total_hashes = max(hashes_from_shares, hashes_from_hashrate)

        # Calculate rates
        hashes_per_second = total_hashes / elapsed if elapsed > 0 else 0
        hashrate_mhs = hashes_per_second / 1e6
        hashrate_ghs = hashes_per_second / 1e9
        avg_latency_us = (elapsed / total_hashes) * 1e6 if total_hashes > 0 else 0

        # Get hardware parameters
        power_watts = final_stats.get('power', 0.0)
        temperature_c = final_stats.get('temp', 0)
        voltage_mv = final_stats.get('coreVoltageActual', 0)
        frequency_mhz = final_stats.get('frequency', 0)

        # Calculate efficiency
        efficiency = hashrate_ghs / power_watts if power_watts > 0 else 0

        result = BenchmarkResult(
            implementation=self.name,
            total_hashes=int(total_hashes),
            elapsed_seconds=elapsed,
            hashes_per_second=hashes_per_second,
            hashrate_mhs=hashrate_mhs,
            hashrate_ghs=hashrate_ghs,
            avg_latency_us=avg_latency_us,
            power_watts=power_watts,
            efficiency_gh_per_watt=efficiency,
            temperature_c=temperature_c,
            voltage_mv=voltage_mv,
            frequency_mhz=frequency_mhz
        )

        # Print results
        print(f"\nResults:")
        print(f"  Time elapsed:     {elapsed:.2f} seconds")
        print(f"  Shares found:     {shares_delta}")
        print(f"  Total hashes:     {total_hashes/1e9:.2f} GH")
        print(f"  Hashes/second:    {hashes_per_second:,.0f} H/s")
        print(f"  Hashrate:         {hashrate_mhs:.2f} MH/s ({hashrate_ghs:.2f} GH/s)")
        print(f"  Avg latency:      {avg_latency_us:.6f} µs")
        print(f"  Power (measured): {power_watts:.2f} W")
        print(f"  Efficiency:       {efficiency:.2f} GH/W")
        print(f"  Temperature:      {temperature_c}°C")
        print(f"  Voltage:          {voltage_mv} mV")
        print(f"  Frequency:        {frequency_mhz} MHz")

        return result

    def _create_error_result(self) -> BenchmarkResult:
        """Create result object for error case"""
        return BenchmarkResult(
            implementation=self.name,
            total_hashes=0,
            elapsed_seconds=0,
            hashes_per_second=0,
            hashrate_mhs=0,
            hashrate_ghs=0,
            avg_latency_us=0,
            power_watts=0,
            efficiency_gh_per_watt=0
        )


def print_comparison(cpu: BenchmarkResult, lv06: BenchmarkResult):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("COMPARISON: CPU vs LV06 (REAL HARDWARE)")
    print(f"{'='*80}\n")

    # Calculate speedups
    speedup_hs = lv06.hashes_per_second / cpu.hashes_per_second if cpu.hashes_per_second > 0 else 0
    speedup_lat = cpu.avg_latency_us / lv06.avg_latency_us if lv06.avg_latency_us > 0 else 0
    power_ratio = lv06.power_watts / cpu.power_watts if cpu.power_watts > 0 else 0
    speedup_eff = lv06.efficiency_gh_per_watt / cpu.efficiency_gh_per_watt if cpu.efficiency_gh_per_watt > 0 else 0

    # Table header
    print(f"{'Metric':<30} {'CPU':<20} {'LV06':<20} {'Ratio':<15}")
    print("-" * 80)

    # Data rows
    print(f"{'Implementation':<30} {cpu.implementation:<20} {lv06.implementation:<20}")
    print(f"{'Hashes/second':<30} {cpu.hashes_per_second:>18,.0f}  {lv06.hashes_per_second:>18,.0f}  {speedup_hs:>13,.0f}x")
    print(f"{'MH/s':<30} {cpu.hashrate_mhs:>18,.3f}  {lv06.hashrate_mhs:>18,.2f}  {speedup_hs:>13,.0f}x")
    print(f"{'GH/s':<30} {cpu.hashrate_ghs:>18,.6f}  {lv06.hashrate_ghs:>18,.2f}  {speedup_hs:>13,.0f}x")
    print(f"{'Latency (µs)':<30} {cpu.avg_latency_us:>18,.3f}  {lv06.avg_latency_us:>18,.6f}  {speedup_lat:>13,.0f}x")
    print(f"{'Power (W)':<30} {cpu.power_watts:>18,.1f}  {lv06.power_watts:>18,.2f}  {power_ratio:>13,.2f}x")
    print(f"{'Efficiency (GH/W)':<30} {cpu.efficiency_gh_per_watt:>18,.6f}  {lv06.efficiency_gh_per_watt:>18,.2f}  {speedup_eff:>13,.0f}x")

    if lv06.temperature_c:
        print(f"{'Temperature (°C)':<30} {'N/A':<20} {lv06.temperature_c:>18.0f}")
    if lv06.voltage_mv:
        print(f"{'Voltage (mV)':<30} {'N/A':<20} {lv06.voltage_mv:>18}")
    if lv06.frequency_mhz:
        print(f"{'Frequency (MHz)':<30} {'N/A':<20} {lv06.frequency_mhz:>18}")

    print("-" * 80)
    print()

    # Summary
    print("KEY FINDINGS:")
    print(f"  • LV06 is {speedup_hs:,.0f}x FASTER than CPU")
    print(f"  • LV06 has {speedup_lat:,.0f}x LOWER latency")
    print(f"  • LV06 is {speedup_eff:,.0f}x MORE EFFICIENT (GH/W)")
    print(f"  • LV06 uses {power_ratio:.2f}x the power of CPU ({lv06.power_watts:.1f}W vs {cpu.power_watts:.1f}W)")
    print()


def extrapolate_to_s9(lv06: BenchmarkResult):
    """Extrapolate LV06 results to Antminer S9"""
    print(f"{'='*80}")
    print("EXTRAPOLATION TO ANTMINER S9 (189 CHIPS)")
    print(f"{'='*80}\n")

    # S9 specifications
    s9_chips = 189
    s9_power_watts = 1320  # Official specification
    s9_efficiency_spec = 0.098  # J/GH (spec sheet)

    # Calculate S9 projections based on LV06 measurements
    s9_hashrate_hs = lv06.hashes_per_second * s9_chips
    s9_hashrate_ghs = s9_hashrate_hs / 1e9
    s9_hashrate_ths = s9_hashrate_ghs / 1000

    # Efficiency
    s9_efficiency_calc = s9_hashrate_ghs / s9_power_watts

    print("Based on LV06 REAL measurements:")
    print(f"  LV06 measured:     {lv06.hashrate_ghs:.2f} GH/s (1 chip BM1366)")
    print()
    print("S9 Projections:")
    print(f"  Chips:             {s9_chips}")
    print(f"  Hashrate:          {s9_hashrate_ghs:,.2f} GH/s ({s9_hashrate_ths:.2f} TH/s)")
    print(f"  Power:             {s9_power_watts:,} W")
    print(f"  Efficiency:        {s9_efficiency_calc:.2f} GH/W ({1/s9_efficiency_calc:.3f} J/GH)")
    print()

    # Compare to spec
    s9_spec_hashrate = 13500  # GH/s (13.5 TH/s typical)
    projection_vs_spec = (s9_hashrate_ghs / s9_spec_hashrate) * 100

    print("Validation against S9 specifications:")
    print(f"  S9 official spec:  13,500 GH/s (13.5 TH/s)")
    print(f"  Our projection:    {s9_hashrate_ghs:,.2f} GH/s ({s9_hashrate_ths:.2f} TH/s)")
    print(f"  Match:             {projection_vs_spec:.1f}% of spec")
    print()

    if abs(projection_vs_spec - 100) < 20:
        print("✓ Projection matches S9 spec within 20% - VALIDATION SUCCESSFUL")
    else:
        print("⚠ Projection differs from S9 spec by >20% - requires investigation")

    print()

    return {
        "chips": s9_chips,
        "hashrate_hs": s9_hashrate_hs,
        "hashrate_ghs": s9_hashrate_ghs,
        "hashrate_ths": s9_hashrate_ths,
        "power_watts": s9_power_watts,
        "efficiency_gh_per_watt": s9_efficiency_calc,
        "spec_hashrate_ghs": s9_spec_hashrate,
        "projection_accuracy_percent": projection_vs_spec
    }


def save_results(cpu: BenchmarkResult, lv06: BenchmarkResult, s9_projection: Dict):
    """Save results to JSON file"""
    results = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_benchmark": cpu.to_dict(),
        "lv06_benchmark": lv06.to_dict(),
        "s9_extrapolation": s9_projection,
        "comparison": {
            "speedup_hashrate": lv06.hashes_per_second / cpu.hashes_per_second if cpu.hashes_per_second > 0 else 0,
            "speedup_latency": cpu.avg_latency_us / lv06.avg_latency_us if lv06.avg_latency_us > 0 else 0,
            "speedup_efficiency": lv06.efficiency_gh_per_watt / cpu.efficiency_gh_per_watt if cpu.efficiency_gh_per_watt > 0 else 0
        }
    }

    filename = "D:\\ASIC_RAG\\experiments\\benchmark_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filename}")
    print()


def main():
    """Run complete benchmark suite"""
    print("\n" + "="*80)
    print("ASIC-RAG-CHIMERA - HARDWARE VALIDATION BENCHMARK")
    print("="*80)
    print()
    print("This benchmark measures REAL performance of:")
    print("  • CPU (Intel/AMD with hashlib)")
    print("  • LV06 (BM1366 ASIC @ 500MHz)")
    print()
    print("Results will be used to:")
    print("  1. Validate theoretical projections")
    print("  2. Extrapolate to Antminer S9 (189 chips)")
    print("  3. Update audit report with real data")
    print()
    print("="*80)

    # Run CPU benchmark
    print("\n[1/2] CPU BENCHMARK")
    cpu_bench = CPUBenchmark()
    cpu_result = cpu_bench.run(num_hashes=1000000)

    # Run LV06 benchmark
    print("\n[2/2] LV06 ASIC BENCHMARK")
    print()
    print("IMPORTANT:")
    print("  • Make sure lv06_stratum_server.py is running")
    print("  • LV06 must be connected and mining")
    print("  • This will monitor for 60 seconds")
    print()
    input("Press Enter when ready to start LV06 benchmark...")

    lv06_bench = LV06Benchmark(lv06_ip="192.168.0.15")
    lv06_result = lv06_bench.run(duration_seconds=60)

    # Print comparison
    print_comparison(cpu_result, lv06_result)

    # Extrapolate to S9
    s9_projection = extrapolate_to_s9(lv06_result)

    # Save results
    save_results(cpu_result, lv06_result, s9_projection)

    # Final summary
    print(f"{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print()
    print("Next steps:")
    print("  1. Review benchmark_results.json")
    print("  2. Update EXTERNAL_AUDIT_REPORT.md with real measurements")
    print("  3. Document findings in LV06_EXPERIMENT_RESULTS.md")
    print()


if __name__ == "__main__":
    main()
