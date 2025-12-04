#!/usr/bin/env python3
"""
ASIC-RAG-CHIMERA Command Line Interface

Commands:
    init        Initialize a new storage
    ingest      Ingest documents
    query       Query the system
    search      Search by keywords
    stats       Show statistics
    verify      Verify block integrity
    benchmark   Run benchmarks

Usage:
    python cli.py init --path ./data
    python cli.py ingest document.txt --category finance
    python cli.py query "What was the Q3 revenue?"
"""

import argparse
import os
import sys
import json
import getpass
import hashlib
from pathlib import Path


def get_master_key(args) -> bytes:
    """Get or generate master key."""
    if args.key_file:
        with open(args.key_file, 'rb') as f:
            return f.read(32)
    elif args.key:
        # Hash provided password to get 32-byte key
        return hashlib.sha256(args.key.encode()).digest()
    elif os.environ.get('ASIC_RAG_KEY'):
        return hashlib.sha256(os.environ['ASIC_RAG_KEY'].encode()).digest()
    else:
        # Prompt for password
        password = getpass.getpass("Enter master password: ")
        return hashlib.sha256(password.encode()).digest()


def cmd_init(args):
    """Initialize storage."""
    from asic_rag_chimera import ASICRAGSystem, SystemConfig
    
    print(f"Initializing ASIC-RAG-CHIMERA at: {args.path}")
    
    path = Path(args.path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Generate and save key if requested
    if args.generate_key:
        key = os.urandom(32)
        key_file = path / ".master_key"
        with open(key_file, 'wb') as f:
            f.write(key)
        print(f"  Master key saved to: {key_file}")
        print("  WARNING: Keep this key secure!")
    
    # Create system to verify initialization
    config = SystemConfig(storage_path=str(path), use_mock_llm=True)
    system = ASICRAGSystem(str(path), config=config)
    system.close()
    
    print("  Initialization complete!")


def cmd_ingest(args):
    """Ingest documents."""
    from asic_rag_chimera import ASICRAGSystem
    
    master_key = get_master_key(args)
    system = ASICRAGSystem(args.storage, master_key=master_key)
    
    try:
        if args.text:
            # Ingest text directly
            print(f"Ingesting text content...")
            block_ids = system.ingest_text(
                args.text,
                source="cli_input",
                category=args.category,
                tags=args.tags.split(',') if args.tags else None
            )
        else:
            # Ingest file
            for file_path in args.files:
                print(f"Ingesting: {file_path}")
                block_ids = system.ingest_file(
                    file_path,
                    tags=args.tags.split(',') if args.tags else None
                )
                print(f"  Created {len(block_ids)} block(s)")
        
        print(f"\nIngestion complete!")
        stats = system.get_statistics()
        print(f"  Total blocks: {stats['total_blocks']}")
        
    finally:
        system.close()


def cmd_query(args):
    """Query the system."""
    from asic_rag_chimera import ASICRAGSystem, SystemConfig
    
    master_key = get_master_key(args)
    
    config = SystemConfig(use_mock_llm=args.mock_llm)
    system = ASICRAGSystem(args.storage, master_key=master_key, config=config)
    
    try:
        print(f"\nQuery: {args.question}")
        print("-" * 50)
        
        result = system.query(
            args.question,
            category=args.category,
            max_results=args.max_results
        )
        
        print(f"\nAnswer: {result.answer}")
        print(f"\nSources ({len(result.retrieved_blocks)}):")
        
        for i, block in enumerate(result.retrieved_blocks[:5]):
            print(f"  [{i+1}] {block.metadata.source} (score: {block.relevance_score:.3f})")
            if args.verbose:
                print(f"      {block.content[:100]}...")
        
        print(f"\nSearch time: {result.search_time_ms:.2f} ms")
        print(f"Total time: {result.total_time_ms:.2f} ms")
        
        if args.json:
            print(f"\nJSON Output:")
            print(json.dumps(result.to_dict(), indent=2))
        
    finally:
        system.close()


def cmd_search(args):
    """Search by keywords."""
    from asic_rag_chimera import ASICRAGSystem
    
    master_key = get_master_key(args)
    system = ASICRAGSystem(args.storage, master_key=master_key)
    
    try:
        keywords = args.keywords.split(',')
        print(f"Searching for: {keywords}")
        print(f"Operation: {args.operation}")
        
        block_ids = system.search(
            keywords,
            operation=args.operation,
            limit=args.limit
        )
        
        print(f"\nFound {len(block_ids)} matching block(s)")
        
        if args.show_content:
            for block_id in block_ids[:10]:
                block = system.get_block(block_id)
                if block:
                    print(f"\n  Block {block_id}:")
                    print(f"    Source: {block['metadata']['source']}")
                    print(f"    Content: {block['content'][:100]}...")
        else:
            print(f"  Block IDs: {block_ids}")
        
    finally:
        system.close()


def cmd_stats(args):
    """Show system statistics."""
    from asic_rag_chimera import ASICRAGSystem
    
    master_key = get_master_key(args)
    system = ASICRAGSystem(args.storage, master_key=master_key)
    
    try:
        stats = system.get_statistics()
        
        print("\nASIC-RAG-CHIMERA Statistics")
        print("=" * 40)
        print(f"  Documents ingested: {stats['documents_ingested']}")
        print(f"  Queries processed: {stats['queries_processed']}")
        print(f"  Total blocks: {stats['total_blocks']}")
        print(f"  Merkle tree leaves: {stats['merkle_leaves']}")
        print(f"\n  Storage operations:")
        print(f"    Reads: {stats['storage_stats']['read_ops']}")
        print(f"    Writes: {stats['storage_stats']['write_ops']}")
        print(f"\n  Index:")
        for key, value in stats['index_size'].items():
            print(f"    {key}: {value}")
        
        if args.json:
            print(f"\nJSON Output:")
            print(json.dumps(stats, indent=2, default=str))
        
    finally:
        system.close()


def cmd_verify(args):
    """Verify block integrity."""
    from asic_rag_chimera import ASICRAGSystem
    
    master_key = get_master_key(args)
    system = ASICRAGSystem(args.storage, master_key=master_key)
    
    try:
        if args.block_id is not None:
            # Verify specific block
            is_valid = system.verify_block(args.block_id)
            print(f"Block {args.block_id}: {'✓ Valid' if is_valid else '✗ Invalid'}")
        else:
            # Verify all blocks
            stats = system.get_statistics()
            total = stats['total_blocks']
            valid = 0
            invalid = 0
            
            print(f"Verifying {total} blocks...")
            
            for block_id in range(total):
                if system.verify_block(block_id):
                    valid += 1
                else:
                    invalid += 1
                    print(f"  ✗ Block {block_id} invalid!")
            
            print(f"\nResults: {valid} valid, {invalid} invalid")
        
    finally:
        system.close()


def cmd_benchmark(args):
    """Run benchmarks."""
    from benchmarks import BenchmarkRunner
    
    runner = BenchmarkRunner(output_dir=args.output)
    
    runner.run_all(
        hash_iterations=args.hash_iterations,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        encryption_iterations=args.encryption_iterations
    )
    
    runner.generate_report()
    
    if args.markdown:
        runner.generate_markdown_report()


def main():
    parser = argparse.ArgumentParser(
        description="ASIC-RAG-CHIMERA Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--storage', '-s', default='./data',
                       help='Storage path (default: ./data)')
    parser.add_argument('--key', '-k', help='Master key password')
    parser.add_argument('--key-file', help='Master key file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init command
    init_parser = subparsers.add_parser('init', help='Initialize storage')
    init_parser.add_argument('--path', default='./data', help='Storage path')
    init_parser.add_argument('--generate-key', action='store_true',
                            help='Generate and save master key')
    
    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('files', nargs='*', help='Files to ingest')
    ingest_parser.add_argument('--text', '-t', help='Text content to ingest')
    ingest_parser.add_argument('--category', '-c', default='general',
                              help='Document category')
    ingest_parser.add_argument('--tags', help='Comma-separated tags')
    
    # query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--category', '-c', help='Filter by category')
    query_parser.add_argument('--max-results', '-n', type=int, default=10,
                             help='Maximum results')
    query_parser.add_argument('--mock-llm', action='store_true',
                             help='Use mock LLM')
    query_parser.add_argument('--json', action='store_true',
                             help='Output as JSON')
    
    # search command
    search_parser = subparsers.add_parser('search', help='Search by keywords')
    search_parser.add_argument('keywords', help='Comma-separated keywords')
    search_parser.add_argument('--operation', '-o', default='OR',
                              choices=['AND', 'OR'], help='Search operation')
    search_parser.add_argument('--limit', '-l', type=int, default=10,
                              help='Maximum results')
    search_parser.add_argument('--show-content', action='store_true',
                              help='Show block content')
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--json', action='store_true',
                             help='Output as JSON')
    
    # verify command
    verify_parser = subparsers.add_parser('verify', help='Verify integrity')
    verify_parser.add_argument('--block-id', '-b', type=int,
                              help='Specific block to verify')
    
    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--output', '-o', default='results',
                             help='Output directory')
    bench_parser.add_argument('--hash-iterations', type=int, default=10000)
    bench_parser.add_argument('--num-docs', type=int, default=10000)
    bench_parser.add_argument('--num-queries', type=int, default=1000)
    bench_parser.add_argument('--encryption-iterations', type=int, default=1000)
    bench_parser.add_argument('--markdown', action='store_true',
                             help='Generate markdown report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch command
    commands = {
        'init': cmd_init,
        'ingest': cmd_ingest,
        'query': cmd_query,
        'search': cmd_search,
        'stats': cmd_stats,
        'verify': cmd_verify,
        'benchmark': cmd_benchmark,
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
