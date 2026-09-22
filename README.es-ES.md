

# ASIC-RAG-CHIMERA

**Simulación en GPU de un motor de hash SHA-256 inspirado en los ASIC de minería de Bitcoin, integrado en una pipeline RAG. Software puro; no se requiere hardware ASIC real.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17872052.svg)](https://doi.org/10.5281/zenodo.17872052)
[![PyPI](https://img.shields.io/pypi/v/asic-rag-chimera.svg)](https://pypi.org/project/asic-rag-chimera/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-57%25-yellow)](coverage.xml)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live_Demo-yellow)](https://huggingface.co/spaces/Agnuxo/ASIC-RAG-CHIMERA)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Qué es esto

ASIC-RAG-CHIMERA es un **artefacto de investigación de software**. Consiste en:

1. Un **motor de hash SHA-256 acelerado por GPU** implementado en PyTorch que *simula* el tipo de hash masivo que realizaría un ASIC estilo Bitcoin. Se ejecuta en una GPU CUDA estándar (o con respaldo de CPU). Es una simulación de software, no un ASIC.
2. Una **pipeline RAG criptográfica** que indexa documentos mediante etiquetas SHA-256 en lugar de embeddings de texto plano, cifra bloques con AES-256-GCM y verifica la integridad con un árbol de Merkle.
3. Un **flujo de trabajo de demostración con registros de pacientes sintéticos** que ilustra cómo se podría configurar la pipeline para datos sensibles a la privacidad (ver `ASIC-RAG-HEALTH_Validation/`). Los datos son ficticios. Esto **no** es una herramienta clínica y no debe usarse para la toma de decisiones médicas.

## Qué NO es esto

- **No** es hardware ASIC real. No hay silicio, ni tape-out en Verilog, ni bitstream de FPGA. La palabra "ASIC" en el nombre hace referencia a la *inspiración* arquitectónica para el módulo simulador de GPU (`asic_simulator/`).
- **No** es un dispositivo médico. La demostración de salud utiliza registros sintéticos y es únicamente ilustrativa.
- **No** es un minero de Bitcoin. El motor SHA-256 se utiliza para indexación dirigida por contenido, no para prueba de trabajo (proof-of-work).

## Instalación

```bash
pip install asic-rag-chimera
```

Extras opcionales:

```bash
pip install "asic-rag-chimera[gpu]"      # Asegúrese de que PyTorch con CUDA esté disponible
pip install "asic-rag-chimera[wandb]"    # Seguimiento de experimentos
pip install "asic-rag-chimera[dev]"      # Pruebas, compilación, twine
```

Desde el código fuente:

```bash
git clone https://github.com/Agnuxo1/ASIC-RAG-CHIMERA.git
cd ASIC-RAG-CHIMERA
pip install -e ".[dev]"
```

## Inicio rápido

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

O utilice la fachada integrada:

```python
from asic_rag_chimera import ASICRAGSystem
system = ASICRAGSystem(storage_path="./data", master_key=os.urandom(32))
system.ingest("document.txt")
result = system.query("What is the revenue?")
```

## Arquitectura

```
┌──────────────┐    texto     ┌─────────────┐    hashes de     ┌────────────────────┐
│ Consulta del │────────────▶│  LLM (GPU)  │─────────────────▶│ Motor GPU SHA-256  │
│    usuario   │              └─────────────┘                  │  (asic_simulator)  │
└──────────────┘                                               └──────────┬─────────┘
                                        ▲                                            │
                                        │ bloques descifrados                     │ búsqueda de
                                        ▼                                            │ hash
                              ┌────────────────────────────────────────────────────┐
                              │ Almacenamiento de bloques cifrados (LMDB / AES-256-│
                              │        GCM)                                        │
                              │ Pruebas de integridad de árbol de Merkle           │
                              └────────────────────────────────────────────────────┘
```

## Ejecución de pruebas y cobertura

```bash
pytest tests/ -v                                    # 53/53 pruebas pasan
pytest tests/ --cov=asic_simulator --cov=rag_system --cov=asic_rag_chimera --cov-report=term --cov-report=xml
```

La cobertura de líneas medida en los paquetes principales es del **57%** (1658 sentencias, 706 no cubiertas), escrita en `coverage.xml`. Lecturas anteriores afirmaban "100%"; eso nunca se midió. Las 53 pruebas pasan todas; simplemente no ejercitan cada rama de `keyword_extractor`, `query_engine`, `key_generator`, etc.

## Modelo de seguridad

| Vector de ataque         | RAG tradicional            | ASIC-RAG-CHIMERA              |
|--------------------------|----------------------------|-------------------------------|
| Robo de disco            | Exposición de texto plano  | Bloques cifrados              |
| Inversión de embedding   | Recuperación parcial       | N/A (no se almacenan embeddings)|
| Enumeración de índices   | Grafos de conocimiento expuestos | Etiquetas SHA-256 opacas |
| Captura de claves        | Acceso permanente          | Claves de sesión con TTL de 30 segundos |
| Manipulación de datos    | No detectado               | Verificación de prueba de Merkle |

Las afirmaciones anteriores describen el *diseño*. Este es un prototipo de investigación, no un producto auditado.

## Estructura del repositorio

```
asic_simulator/     Motor GPU SHA-256 + índice de etiquetas + generador de claves
rag_system/         Procesador de documentos, almacenamiento de bloques, motor de consultas
asic_rag_chimera.py Fachada integrada (ASICRAGSystem)
tests/              53 pruebas de pytest
benchmarks/         Microbenchmarks para latencia de hash y búsqueda
archive/            Artefactos históricos (PDFs, HTML, directorios duplicados) — no se incluyen
huggingface_space/  Aplicación de demostración de HF Space
```

## Citación

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

Ver [`CITATION.cff`](CITATION.cff).

## Autor

**Francisco Angulo de Lafuente** — [GitHub @Agnuxo1](https://github.com/Agnuxo1)

## Licencia

MIT — ver [LICENSE](LICENSE).

---

## Proyectos relacionados

Parte del catálogo de código abierto v1.0.0 de [@Agnuxo1](https://github.com/Agnuxo1) (abril de 2026).

**Constelación AgentBoot** — agentes y bucles de investigación
- [AgentBoot](https://github.com/Agnuxo1/AgentBoot) — Agente de IA conversacional para detección de hardware bare-metal e instalación de SO.
- [autoresearch-nano](https://github.com/Agnuxo1/autoresearch) — Bucle de investigación de ML autónomo basado en nanoGPT.
- [The Living Agent](https://github.com/Agnuxo1/The-Living-Agent) — Agente de investigación autónomo tipo tablero de ajedrez 16x16.
- [benchclaw-integrations](https://github.com/Agnuxo1/benchclaw-integrations) — Adaptadores de framework de agentes para la API BenchClaw.

**Constelación CHIMERA / neuromórfica** — computación científica nativa de GPU
- [NeuroCHIMERA](https://github.com/Agnuxo1/NeuroCHIMERA__GPU-Native_Neuromorphic_Consciousness) — Framework neuromórfico nativo de GPU en shaders de cómputo OpenGL.
- [Holographic-Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir) — Computación de reservorio con backend ASIC simulado.
- [QESN-MABe](https://github.com/Agnuxo1/QESN_MABe_V2_REPO) — Red de estado eco inspirada en la mecánica cuántica en una cuadrícula 2D (clásica).
- [ARC2-CHIMERA](https://github.com/Agnuxo1/ARC2_CHIMERA) — PoC de investigación: primitivas OpenGL para razonamiento simbólico.
- [Quantum-GPS](https://github.com/Agnuxo1/Quantum-GPS-Unified-Navigation-System) — Navegador de GPU inspirado en la mecánica cuántica (solucionador Eikonal clásico).
