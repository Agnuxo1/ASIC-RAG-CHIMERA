"""
Hash Performance Benchmark for ASIC-RAG

Measures SHA-256 computation performance across:
- Pure Python implementation
- NumPy vectorized
- ASIC simulator
- CHIMERA GPU acceleration

Usage:
    python -m benchmarks.hash_performance --iterations 10000
"""

import argparse
import hashlib
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    implementation: str
    iterations: int
    total_time_ms: float
    mean_time_us: float
    min_time_us: float
    max_time_us: float
    std_dev_us: float
    hashes_per_second: float
    memory_mb: float


class HashBenchmark:
    """
    Comprehensive SHA-256 hash benchmark.
    
    Compares multiple implementations:
    - hashlib (baseline)
    - ASIC simulator
    - CHIMERA GPU
    
    Example:
        >>> benchmark = HashBenchmark()
        >>> results = benchmark.run_all(iterations=10000)
        >>> benchmark.print_results(results)
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def benchmark_hashlib(
        self,
        iterations: int = 10000,
        data_size: int = 256
    ) -> BenchmarkResult:
        """Benchmark Python hashlib."""
        # Generate test data
        test_data = [os.urandom(data_size) for _ in range(iterations)]
        
        # Warmup
        for d in test_data[:100]:
            hashlib.sha256(d).digest()
        
        # Benchmark
        times = []
        start_total = time.perf_counter()
        
        for d in test_data:
            start = time.perf_counter()
            hashlib.sha256(d).digest()
            times.append((time.perf_counter() - start) * 1e6)  # microseconds
        
        total_time = (time.perf_counter() - start_total) * 1000  # milliseconds
        
        import statistics
        return BenchmarkResult(
            implementation="hashlib (Python)",
            iterations=iterations,
            total_time_ms=total_time,
            mean_time_us=statistics.mean(times),
            min_time_us=min(times),
            max_time_us=max(times),
            std_dev_us=statistics.stdev(times) if len(times) > 1 else 0,
            hashes_per_second=iterations / (total_time / 1000),
            memory_mb=0.0
        )
    
    def benchmark_asic_simulator(
        self,
        iterations: int = 10000,
        data_size: int = 256
    ) -> BenchmarkResult:
        """Benchmark ASIC simulator."""
        from asic_simulator import GPUHashEngine
        
        engine = GPUHashEngine()
        test_data = [os.urandom(data_size) for _ in range(iterations)]
        
        # Warmup
        engine.hash_batch(test_data[:100])
        engine.clear_cache()
        engine.clear_cache()
        # engine.reset_metrics()
        
        # Benchmark
        start_total = time.perf_counter()
        result = engine.hash_batch(test_data)
        total_time = (time.perf_counter() - start_total) * 1000
        
        # Calculate per-hash times
        mean_time_us = (total_time * 1000) / iterations
        
        import statistics
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        return BenchmarkResult(
            implementation="ASIC Simulator",
            iterations=iterations,
            total_time_ms=total_time,
            mean_time_us=mean_time_us,
            min_time_us=mean_time_us, # Approximation
            max_time_us=mean_time_us, # Approximation
            std_dev_us=0.0, # Not available without individual times
            hashes_per_second=result.hashes_per_second,
            memory_mb=memory_mb
        )
    
    def benchmark_chimera_gpu(
        self,
        iterations: int = 10000,
        data_size: int = 256
    ) -> Optional[BenchmarkResult]:
        """Benchmark CHIMERA GPU acceleration."""
        try:
            from chimera_integration import GPUHashEngine
        except ImportError:
            print("CHIMERA integration not available")
            return None
        
        engine = GPUHashEngine()
        
        if not engine.is_gpu_available:
            print("GPU not available for CHIMERA benchmark")
            return None
        
        test_data = [os.urandom(data_size) for _ in range(iterations)]
        
        # Warmup
        engine.hash_batch(test_data[:100])
        
        # Benchmark
        start_total = time.perf_counter()
        result = engine.hash_batch(test_data)
        total_time = (time.perf_counter() - start_total) * 1000
        
        return BenchmarkResult(
            implementation="CHIMERA GPU",
            iterations=iterations,
            total_time_ms=total_time,
            mean_time_us=total_time * 1000 / iterations,
            min_time_us=0,  # Not tracked individually
            max_time_us=0,
            std_dev_us=0,
            hashes_per_second=result.hashes_per_second,
            memory_mb=result.gpu_memory_mb
        )
    
    def benchmark_batch_sizes(
        self,
        batch_sizes: List[int] = None,
        data_size: int = 256
    ) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [100, 1000, 10000, 100000]
        
        results = {
            "hashlib": [],
            "asic_simulator": [],
            "chimera_gpu": []
        }
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            results["hashlib"].append(
                self.benchmark_hashlib(batch_size, data_size)
            )
            
            results["asic_simulator"].append(
                self.benchmark_asic_simulator(batch_size, data_size)
            )
            
            gpu_result = self.benchmark_chimera_gpu(batch_size, data_size)
            if gpu_result:
                results["chimera_gpu"].append(gpu_result)
        
        return results
    
    def run_all(
        self,
        iterations: int = 10000,
        data_size: int = 256
    ) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []
        
        print("Running hashlib benchmark...")
        results.append(self.benchmark_hashlib(iterations, data_size))
        
        print("Running ASIC simulator benchmark...")
        results.append(self.benchmark_asic_simulator(iterations, data_size))
        
        print("Running CHIMERA GPU benchmark...")
        gpu_result = self.benchmark_chimera_gpu(iterations, data_size)
        if gpu_result:
            results.append(gpu_result)
        
        self.results = results
        return results
    
    def print_results(self, results: List[BenchmarkResult] = None):
        """Print benchmark results in table format."""
        results = results or self.results
        
        print("\n" + "=" * 80)
        print("HASH BENCHMARK RESULTS")
        print("=" * 80)
        
        # Header
        print(f"{'Implementation':<25} {'Iterations':>12} {'Total (ms)':>12} "
              f"{'Mean (µs)':>12} {'H/sec':>15}")
        print("-" * 80)
        
        for r in results:
            print(f"{r.implementation:<25} {r.iterations:>12,} {r.total_time_ms:>12.2f} "
                  f"{r.mean_time_us:>12.2f} {r.hashes_per_second:>15,.0f}")
        
        print("-" * 80)
        
        # Speedup comparison
        if len(results) >= 2:
            baseline = results[0].hashes_per_second
            print("\nSpeedup vs baseline (hashlib):")
            for r in results[1:]:
                speedup = r.hashes_per_second / baseline if baseline > 0 else 0
                print(f"  {r.implementation}: {speedup:.2f}x")
    
    def export_results(
        self,
        filepath: str,
        results: List[BenchmarkResult] = None
    ):
        """Export results to JSON file."""
        results = results or self.results
        
        data = {
            "benchmark": "hash_performance",
            "timestamp": time.time(),
            "results": [asdict(r) for r in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Hash Performance Benchmark")
    parser.add_argument("--iterations", type=int, default=10000,
                       help="Number of iterations")
    parser.add_argument("--data-size", type=int, default=256,
                       help="Size of data to hash in bytes")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--batch-sizes", action="store_true",
                       help="Run batch size comparison")
    
    args = parser.parse_args()
    
    benchmark = HashBenchmark()
    
    if args.batch_sizes:
        print("Running batch size comparison...")
        batch_results = benchmark.benchmark_batch_sizes()
        
        print("\n" + "=" * 80)
        print("BATCH SIZE COMPARISON")
        print("=" * 80)
        
        for impl, results in batch_results.items():
            if results:
                print(f"\n{impl}:")
                for r in results:
                    print(f"  {r.iterations:>8}: {r.hashes_per_second:>12,.0f} H/s")
    else:
        results = benchmark.run_all(args.iterations, args.data_size)
        benchmark.print_results(results)
    
    if args.output:
        benchmark.export_results(args.output)
        print(f"\nResults exported to: {args.output}")


if __name__ == "__main__":
    main()
