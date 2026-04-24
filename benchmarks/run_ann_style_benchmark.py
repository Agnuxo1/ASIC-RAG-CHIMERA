"""
ANN-Style Benchmark for ASIC-RAG (Tag Search)

This script adapts the ASIC-RAG tag search performance to comparable metrics
used in Vector Database benchmarks (ANN-Benchmarks):
- QPS (Queries Per Second)
- Latency (p95, p99)
- Recall (Precision of retrieval)

It generates a synthetic dataset of documents with random tags, establishes
a ground truth using brute-force Python sets, and then measures the
performance of the ASIC Simulator's retrieval engine.
"""

import time
import random
import argparse
import json
import statistics
import sys
import os
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from asic_simulator import IndexManager, SearchOperation
except ImportError:
    # Fallback/Mock for standalone testing if asic_simulator is missing
    print("WARNING: asic_simulator not found. Using Mock classes.")
    class SearchOperation:
        AND = "AND"
        OR = "OR"
    
    class IndexManager:
        def __init__(self):
            self.docs = {}
        def add_tag(self, tag, doc_id, category):
            pass
        def search(self, tags, op):
            return type('obj', (object,), {'block_ids': []})

@dataclass
class BenchmarkMetrics:
    dataset_size: int
    query_count: int
    qps: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    recall: float
    index_build_time_s: float

class ANNStyleBenchmark:
    def __init__(self, num_docs: int = 10000, num_tags_universe: int = 1000, tags_per_doc: int = 10):
        self.num_docs = num_docs
        self.num_tags_universe = num_tags_universe
        self.tags_per_doc = tags_per_doc
        
        self.tags_universe = [f"tag_{i:04d}" for i in range(num_tags_universe)]
        self.ground_truth_docs: Dict[int, Set[str]] = {}
        
        self.index_manager = IndexManager()
    
    def generate_and_index(self):
        """Generates synthetic data and populates the ASIC Index."""
        print(f"Generating {self.num_docs} documents with ~{self.tags_per_doc} tags each...")
        
        start_time = time.time()
        
        for doc_id in range(self.num_docs):
            # Select random tags
            doc_tags = random.sample(self.tags_universe, self.tags_per_doc)
            self.ground_truth_docs[doc_id] = set(doc_tags)
            
            # Add to ASIC Index
            # Category is dummy for this benchmark
            for tag in doc_tags:
                self.index_manager.add_tag(tag, doc_id, category="bench_cat")
                
        build_time = time.time() - start_time
        print(f"Indexing complete in {build_time:.2f}s")
        return build_time

    def _get_ground_truth(self, query_tags: List[str], op: str) -> Set[int]:
        """Calculates exact expected document IDs using Python sets."""
        query_set = set(query_tags)
        matches = set()
        
        for doc_id, doc_tags in self.ground_truth_docs.items():
            if op == SearchOperation.AND:
                if query_set.issubset(doc_tags):
                    matches.add(doc_id)
            elif op == SearchOperation.OR:
                if not query_set.isdisjoint(doc_tags):
                    matches.add(doc_id)
        return matches

    def run_benchmark(self, num_queries: int = 1000, tags_per_query: int = 2) -> BenchmarkMetrics:
        print(f"Running {num_queries} queries (tags/query={tags_per_query})...")
        
        latencies = []
        recalls = []
        
        start_total = time.perf_counter()
        
        for _ in range(num_queries):
            # Generate random query
            query_tags = random.sample(self.tags_universe, tags_per_query)
            
            # 1. Get Ground Truth
            # We measure recall against strict AND search for this benchmark
            expected_ids = self._get_ground_truth(query_tags, SearchOperation.AND)
            
            # Skip empty queries to measure meaningful retrieval
            if not expected_ids:
                # We count this as valid but recall is 1.0 if both are empty
                # But for latency we run the search anyway
                pass

            # 2. Run ASIC Search
            t0 = time.perf_counter()
            result = self.index_manager.search(query_tags, SearchOperation.AND)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0) # ms
            
            # 3. Calculate Recall
            retrieved_ids = set(result.block_ids) if hasattr(result, 'block_ids') else set()
            
            if not expected_ids:
                 # If no docs expected, and we found none -> perfect recall
                 # If we found some -> hallucination (but simplistic recall def: intersection / expected)
                 # Standard Recall = TP / (TP + FN). If Denom is 0, undefined.
                 # We'll treat "both empty" as 1.0
                 if not retrieved_ids:
                     recalls.append(1.0)
                 else:
                     recalls.append(0.0)
            else:
                intersection = expected_ids.intersection(retrieved_ids)
                recalls.append(len(intersection) / len(expected_ids))

        total_time_s = time.perf_counter() - start_total
        sorted_latencies = sorted(latencies)
        
        metrics = BenchmarkMetrics(
            dataset_size=self.num_docs,
            query_count=num_queries,
            qps=num_queries / total_time_s,
            mean_latency_ms=statistics.mean(latencies),
            p95_latency_ms=sorted_latencies[int(len(latencies) * 0.95)],
            p99_latency_ms=sorted_latencies[int(len(latencies) * 0.99)],
            recall=statistics.mean(recalls) if recalls else 0.0,
            index_build_time_s=0.0 # Filled later
        )
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Run ANN-Style Benchmark for ASIC-RAG")
    parser.add_argument("--num-docs", type=int, default=10000, help="Number of documents")
    parser.add_argument("--metrics-file", type=str, default="ann_benchmark_results.json", help="Output file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()
    
    # WandB Init
    if not args.no_wandb:
        try:
            import os
            import wandb
            # Login via environment variable (do not hardcode API keys)
            _wandb_key = os.environ.get("WANDB_API_KEY")
            if _wandb_key:
                wandb.login(key=_wandb_key)
            else:
                wandb.login()
            wandb.init(
                project="asic-rag-chimera",
                entity="lareliquia-angulo",
                config={
                    "num_docs": args.num_docs,
                    "benchmark_type": "ann_style_latency"
                }
            )
            print("WandB initialized successfully.")
        except ImportError:
            print("WandB not installed. Skipping logging.")
            args.no_wandb = True
        except Exception as e:
            print(f"WandB init failed: {e}")
            args.no_wandb = True

    benchmark = ANNStyleBenchmark(num_docs=args.num_docs)
    build_time = benchmark.generate_and_index()
    
    metrics = benchmark.run_benchmark()
    metrics.index_build_time_s = build_time
    
    # Log to WandB
    if not args.no_wandb:
        wandb.log({
            "qps": metrics.qps,
            "mean_latency_ms": metrics.mean_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "recall": metrics.recall,
            "index_build_time_s": metrics.index_build_time_s,
            "dataset_size": metrics.dataset_size
        })
        print(f"Metrics logged to WandB run: {wandb.run.get_url()}")
        wandb.finish()

    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS (N={metrics.dataset_size})")
    print("="*60)
    print(f"QPS:             {metrics.qps:,.2f}")
    print(f"Avg Latency:     {metrics.mean_latency_ms:.4f} ms")
    print(f"P95 Latency:     {metrics.p95_latency_ms:.4f} ms")
    print(f"P99 Latency:     {metrics.p99_latency_ms:.4f} ms")
    print(f"Recall:          {metrics.recall:.4f}")
    print("-" * 60)
    
    # Save to file
    with open(args.metrics_file, 'w') as f:
        json.dump(asdict(metrics), f, indent=4)
    print(f"Results saved to {os.path.abspath(args.metrics_file)}")

if __name__ == "__main__":
    main()
