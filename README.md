# ASIC-RAG-CHIMERA

**GPU simulation of a SHA-256 hash engine inspired by Bitcoin mining ASICs, wired into a RAG pipeline. Pure software; no real ASIC hardware required.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17872052.svg)](https://doi.org/10.5281/zenodo.17872052)
[![PyPI](https://img.shields.io/pypi/v/asic-rag-chimera.svg)](https://pypi.org/project/asic-rag-chimera/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-57%25-yellow)](coverage.xml)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live_Demo-yellow)](https://huggingface.co/spaces/Agnuxo/ASIC-RAG-CHIMERA)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What this is

ASIC-RAG-CHIMERA is a **software research artifact**. It consists of:

1. A **GPU-accelerated SHA-256 hash engine** implemented in PyTorch that *simulates* the kind of bulk hashing a Bitcoin-style ASIC would do. It runs on a normal CUDA GPU (or CPU fallback). It is a software simulation, not an ASIC.
2. A **cryptographic RAG pipeline** that indexes documents by SHA-256 tags instead of plaintext embeddings, encrypts blocks with AES-256-GCM, and verifies integrity with a Merkle tree.
3. A **demonstration workflow with synthetic patient records** illustrating how the pipeline could be configured for privacy-sensitive data (see `ASIC-RAG-HEALTH_Validation/`). The data is fabricated. This is **not** a clinical tool and must not be used for medical decision making.

## What this is NOT

- **Not** real ASIC hardware. There is no silicon, no Verilog tape-out, no FPGA bitstream. The "ASIC" in the name refers to the architectural *inspiration* for the GPU simulator module (`asic_simulator/`).
- **Not** a medical device. The health demo uses synthetic records and is illustrative only.
- **Not** a Bitcoin miner. The SHA-256 engine is used for content-addressed indexing, not proof-of-work.

## Installation

```bash
pip install asic-rag-chimera
```

Optional extras:

```bash
pip install "asic-rag-chimera[gpu]"      # Ensure PyTorch with CUDA is available
pip install "asic-rag-chimera[wandb]"    # Experiment tracking
pip install "asic-rag-chimera[dev]"      # Tests, build, twine
```

From source:

```bash
git clone https://github.com/Agnuxo1/ASIC-RAG-CHIMERA.git
cd ASIC-RAG-CHIMERA
pip install -e ".[dev]"
```

## Quick start

```python
import os
from asic_simulator import GPUHashEngine, IndexManager, KeyGenerator
from rag_system import DocumentProcessor, QueryEngine

hash_engine = GPUHashEngine()
index_manager = IndexManager()
key_generator = KeyGenerator(master_key=os.urandom(32))

processor = DocumentProcessor()
blocks = processor.create_blocks("Your document content here")

query_engine = QueryEngine(index_manager, hash_engine)
results = query_engine.search("your query", max_results=5)
```

Or use the integrated facade:

```python
from asic_rag_chimera import ASICRAGSystem
system = ASICRAGSystem(storage_path="./data", master_key=os.urandom(32))
system.ingest("document.txt")
result = system.query("What is the revenue?")
```

## Architecture

```
┌──────────────┐    text     ┌─────────────┐    tag hashes    ┌────────────────────┐
│  User query  │────────────▶│  LLM (GPU)  │─────────────────▶│ GPU SHA-256 engine │
└──────────────┘             └─────────────┘                  │  (asic_simulator)  │
                                    ▲                         └──────────┬─────────┘
                                    │ decrypted blocks                   │ hash lookup
                                    ▼                                    ▼
                         ┌────────────────────────────────────────────────────┐
                         │  Encrypted block storage (LMDB / AES-256-GCM)      │
                         │  Merkle tree integrity proofs                      │
                         └────────────────────────────────────────────────────┘
```

## Running tests and coverage

```bash
pytest tests/ -v                                    # 53/53 tests pass
pytest tests/ --cov=asic_simulator --cov=rag_system --cov=asic_rag_chimera --cov-report=term --cov-report=xml
```

Measured line coverage on the core packages is **57%** (1658 statements, 706 missed), written to `coverage.xml`. Previous READMEs claimed "100%" — that was never measured. The 53 tests all pass; they simply don't exercise every branch of `keyword_extractor`, `query_engine`, `key_generator`, etc.

## Security model

| Attack vector            | Traditional RAG          | ASIC-RAG-CHIMERA            |
|--------------------------|--------------------------|-----------------------------|
| Disk theft               | Plaintext exposure       | Encrypted blocks            |
| Embedding inversion      | Partial recovery         | N/A (no embeddings stored)  |
| Index enumeration        | Knowledge graph exposed  | Opaque SHA-256 tags         |
| Key capture              | Permanent access         | 30-second TTL session keys  |
| Data tampering           | Undetected               | Merkle proof verification   |

Claims above describe the *design*. This is a research prototype, not an audited product.

## Repository layout

```
asic_simulator/     GPU SHA-256 engine + tag index + key generator
rag_system/         Document processor, block storage, query engine
asic_rag_chimera.py Integrated facade (ASICRAGSystem)
tests/              53 pytest tests
benchmarks/         Microbenchmarks for hash and search latency
archive/            Historical artefacts (PDFs, HTML, duplicate dirs) — not shipped
huggingface_space/  HF Space demo app
```

## Citation

```bibtex
@software{angulo_asic_rag_chimera_2026,
  author  = {Angulo de Lafuente, Francisco},
  title   = {ASIC-RAG-CHIMERA: GPU Simulation of a SHA-256 Hash Engine for Cryptographic RAG},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.17872052},
  url     = {https://github.com/Agnuxo1/ASIC-RAG-CHIMERA}
}
```

See [`CITATION.cff`](CITATION.cff).

## Author

**Francisco Angulo de Lafuente** — [GitHub @Agnuxo1](https://github.com/Agnuxo1)

## License

MIT — see [LICENSE](LICENSE).

---

## Related projects

Part of the [@Agnuxo1](https://github.com/Agnuxo1) v1.0.0 open-source catalog (April 2026).

**AgentBoot constellation** — agents and research loops
- [AgentBoot](https://github.com/Agnuxo1/AgentBoot) — Conversational AI agent for bare-metal hardware detection and OS install.
- [autoresearch-nano](https://github.com/Agnuxo1/autoresearch) — nanoGPT-based autonomous ML research loop.
- [The Living Agent](https://github.com/Agnuxo1/The-Living-Agent) — 16x16 Chess-Grid autonomous research agent.
- [benchclaw-integrations](https://github.com/Agnuxo1/benchclaw-integrations) — Agent-framework adapters for the BenchClaw API.

**CHIMERA / neuromorphic constellation** — GPU-native scientific computing
- [NeuroCHIMERA](https://github.com/Agnuxo1/NeuroCHIMERA__GPU-Native_Neuromorphic_Consciousness) — GPU-native neuromorphic framework on OpenGL compute shaders.
- [Holographic-Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir) — Reservoir computing with simulated ASIC backend.
- [QESN-MABe](https://github.com/Agnuxo1/QESN_MABe_V2_REPO) — Quantum-inspired Echo State Network on a 2D lattice (classical).
- [ARC2-CHIMERA](https://github.com/Agnuxo1/ARC2_CHIMERA) — Research PoC: OpenGL primitives for symbolic reasoning.
- [Quantum-GPS](https://github.com/Agnuxo1/Quantum-GPS-Unified-Navigation-System) — Quantum-inspired GPU navigator (classical Eikonal solver).
