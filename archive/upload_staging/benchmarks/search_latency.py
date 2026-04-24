"""
Search Latency Benchmark for ASIC-RAG

Measures query performance metrics:
- Tag lookup latency
- Merkle verification time
- Block retrieval time
- Full query latency

Usage:
    python -m benchmarks.search_latency --num-docs 10000 --num-queries 1000
"""

import argparse
import hashlib
import time
import json
import os
import sys
import random
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SearchBenchmarkResult:
    """Result of search benchmark."""
    operation: str
    num_documents: int
    num_queries: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    queries_per_second: float


class SearchBenchmark:
    """
    Search latency benchmark for ASIC-RAG system.
    
    Measures:
    - Index lookup speed
    - Merkle verification overhead
    - Block retrieval latency
    - End-to-end query time
    
    Example:
        >>> benchmark = SearchBenchmark()
        >>> benchmark.setup(num_docs=10000)
        >>> results = benchmark.run_all()
        >>> benchmark.print_results(results)
    """
    
    def __init__(self):
        self.index_manager = None
        self.block_storage = None
        self.merkle_tree = None
        self.test_tags = []
        self.results: List[SearchBenchmarkResult] = []
        self._temp_dir = None
    
    def setup(
        self,
        num_docs: int = 10000,
        tags_per_doc: int = 10,
        num_categories: int = 5
    ):
        """
        Set up test data for benchmarks.
        
        Args:
            num_docs: Number of test documents
            tags_per_doc: Average tags per document
            num_categories: Number of categories
        """
        print(f"Setting up benchmark with {num_docs} documents...")
        
        from asic_simulator import IndexManager, MerkleTree, GPUHashEngine
        
        self.index_manager = IndexManager()
        self.hash_engine = GPUHashEngine()
        self.merkle_tree = MerkleTree(self.hash_engine)
        
        # Generate categories
        categories = [f"category_{i}" for i in range(num_categories)]
        
        # Common tags for testing
        common_tags = [f"tag_{i}" for i in range(100)]
        
        # Create test documents
        for doc_id in range(num_docs):
            # Assign random category
            category = random.choice(categories)
            
            # Generate tags
            doc_tags = random.sample(common_tags, min(tags_per_doc, len(common_tags)))
            doc_tags.append(category)
            
            # Add to index
            for tag in doc_tags:
                self.index_manager.add_tag(tag, doc_id, category)
            
            # Add to Merkle tree
            doc_hash = hashlib.sha256(f"doc_{doc_id}".encode()).digest()
            self.merkle_tree.add_leaf(doc_hash)
        
        # Build Merkle tree
        self.merkle_tree.build()
        
        # Store test tags for queries
        self.test_tags = common_tags + categories
        
        print(f"  Index size: {self.index_manager.get_statistics()['total_tags']} tags")
        print(f"  Merkle tree: {len(self.merkle_tree.leaves)} leaves")
    
    def benchmark_tag_lookup(
        self,
        num_queries: int = 1000
    ) -> SearchBenchmarkResult:
        """Benchmark single tag lookup."""
        times = []
        
        for _ in range(num_queries):
            tag = random.choice(self.test_tags)
            
            start = time.perf_counter()
            self.index_manager.get_blocks_for_tag(tag)
            times.append((time.perf_counter() - start) * 1000)
        
        return self._create_result("Tag Lookup", times, num_queries)
    
    def benchmark_and_search(
        self,
        num_queries: int = 1000,
        tags_per_query: int = 3
    ) -> SearchBenchmarkResult:
        """Benchmark AND search with multiple tags."""
        from asic_simulator import SearchOperation
        
        times = []
        
        for _ in range(num_queries):
            tags = random.sample(self.test_tags, min(tags_per_query, len(self.test_tags)))
            
            start = time.perf_counter()
            self.index_manager.search(tags, SearchOperation.AND)
            times.append((time.perf_counter() - start) * 1000)
        
        return self._create_result(f"AND Search ({tags_per_query} tags)", times, num_queries)
    
    def benchmark_or_search(
        self,
        num_queries: int = 1000,
        tags_per_query: int = 3
    ) -> SearchBenchmarkResult:
        """Benchmark OR search with multiple tags."""
        from asic_simulator import SearchOperation
        
        times = []
        
        for _ in range(num_queries):
            tags = random.sample(self.test_tags, min(tags_per_query, len(self.test_tags)))
            
            start = time.perf_counter()
            self.index_manager.search(tags, SearchOperation.OR)
            times.append((time.perf_counter() - start) * 1000)
        
        return self._create_result(f"OR Search ({tags_per_query} tags)", times, num_queries)
    
    def benchmark_merkle_verification(
        self,
        num_queries: int = 1000
    ) -> SearchBenchmarkResult:
        """Benchmark Merkle proof generation and verification."""
        times = []
        
        for _ in range(num_queries):
            leaf_index = random.randint(0, len(self.merkle_tree.leaves) - 1)
            
            start = time.perf_counter()
            proof = self.merkle_tree.get_proof(leaf_index)
            self.merkle_tree.verify_proof(proof)
            times.append((time.perf_counter() - start) * 1000)
        
        return self._create_result("Merkle Verification", times, num_queries)
    
    def benchmark_full_query(
        self,
        num_queries: int = 1000,
        tags_per_query: int = 2
    ) -> SearchBenchmarkResult:
        """Benchmark full query pipeline (search + verify)."""
        from asic_simulator import SearchOperation
        
        times = []
        
        for _ in range(num_queries):
            tags = random.sample(self.test_tags, min(tags_per_query, len(self.test_tags)))
            
            start = time.perf_counter()
            
            # Search
            result = self.index_manager.search(tags, SearchOperation.AND)
            
            # Verify first result
            if result.block_ids:
                block_id = result.block_ids[0]
                if block_id < len(self.merkle_tree.leaves):
                    proof = self.merkle_tree.get_proof(block_id)
                    self.merkle_tree.verify_proof(proof)
            
            times.append((time.perf_counter() - start) * 1000)
        
        return self._create_result("Full Query Pipeline", times, num_queries)
    
    def _create_result(
        self,
        operation: str,
        times: List[float],
        num_queries: int
    ) -> SearchBenchmarkResult:
        """Create benchmark result from times list."""
        sorted_times = sorted(times)
        
        return SearchBenchmarkResult(
            operation=operation,
            num_documents=self.index_manager.get_statistics()['total_blocks'] if self.index_manager else 0,
            num_queries=num_queries,
            mean_latency_ms=statistics.mean(times),
            p50_latency_ms=sorted_times[int(len(sorted_times) * 0.5)],
            p95_latency_ms=sorted_times[int(len(sorted_times) * 0.95)],
            p99_latency_ms=sorted_times[int(len(sorted_times) * 0.99)],
            min_latency_ms=min(times),
            max_latency_ms=max(times),
            queries_per_second=num_queries / (sum(times) / 1000) if sum(times) > 0 else 0
        )
    
    def run_all(
        self,
        num_queries: int = 1000
    ) -> List[SearchBenchmarkResult]:
        """Run all search benchmarks."""
        results = []
        
        print("Running tag lookup benchmark...")
        results.append(self.benchmark_tag_lookup(num_queries))
        
        print("Running AND search benchmark...")
        results.append(self.benchmark_and_search(num_queries))
        
        print("Running OR search benchmark...")
        results.append(self.benchmark_or_search(num_queries))
        
        print("Running Merkle verification benchmark...")
        results.append(self.benchmark_merkle_verification(num_queries))
        
        print("Running full query benchmark...")
        results.append(self.benchmark_full_query(num_queries))
        
        self.results = results
        return results
    
    def print_results(self, results: List[SearchBenchmarkResult] = None):
        """Print benchmark results."""
        results = results or self.results
        
        print("\n" + "=" * 100)
        print("SEARCH LATENCY BENCHMARK RESULTS")
        print("=" * 100)
        
        print(f"{'Operation':<30} {'Mean (ms)':>12} {'P50 (ms)':>12} "
              f"{'P95 (ms)':>12} {'P99 (ms)':>12} {'QPS':>12}")
        print("-" * 100)
        
        for r in results:
            print(f"{r.operation:<30} {r.mean_latency_ms:>12.4f} {r.p50_latency_ms:>12.4f} "
                  f"{r.p95_latency_ms:>12.4f} {r.p99_latency_ms:>12.4f} "
                  f"{r.queries_per_second:>12,.0f}")
        
        print("-" * 100)
    
    def export_results(
        self,
        filepath: str,
        results: List[SearchBenchmarkResult] = None
    ):
        """Export results to JSON."""
        results = results or self.results
        
        data = {
            "benchmark": "search_latency",
            "timestamp": time.time(),
            "results": [asdict(r) for r in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Search Latency Benchmark")
    parser.add_argument("--num-docs", type=int, default=10000,
                       help="Number of documents to index")
    parser.add_argument("--num-queries", type=int, default=1000,
                       help="Number of queries to run")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    benchmark = SearchBenchmark()
    benchmark.setup(num_docs=args.num_docs)
    
    results = benchmark.run_all(num_queries=args.num_queries)
    benchmark.print_results(results)
    
    if args.output:
        benchmark.export_results(args.output)
        print(f"\nResults exported to: {args.output}")


if __name__ == "__main__":
    main()