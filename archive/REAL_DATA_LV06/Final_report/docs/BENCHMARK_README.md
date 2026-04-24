# CHIMERA Definitive Validation Experiment

## Purpose

This experiment provides **rigorous, unbiased validation** of all claims made in the ASIC-RAG-CHIMERA paper. Each claim receives a clear PASS/FAIL verdict based on proper statistical tests.

## Prerequisites

1. `chronos_bridge_v2.py` running on your PC
2. Lucky Miner LV06 connected and configured to mine to the bridge
3. Python 3.8+ with no external dependencies (uses only stdlib)

## Usage

```bash
# Default: 10 minute sampling (recommended for statistical validity)
python chimera_definitive_validation.py

# Custom duration (seconds)
python chimera_definitive_validation.py 300   # 5 minutes
python chimera_definitive_validation.py 1800  # 30 minutes (most rigorous)
```

## Tests Performed

### Phase 1: Basic Statistics
| Test | What It Measures | Pass Criteria |
|------|------------------|---------------|
| Throughput QPS | Events per second | ~0.18 QPS (within reasonable range) |
| Mean Latency | Average inter-arrival time | ~6188ms (within 50% of claim) |

### Phase 2: Entropy & Randomness (NIST SP 800-22 Inspired)
| Test | What It Measures | Pass Criteria |
|------|------------------|---------------|
| Shannon Entropy | Bits of randomness per symbol | ≥3.0 bits for crypto use |
| Monobit Test | Uniform distribution | p-value ≥ 0.01 |
| Runs Test | Absence of patterns | p-value ≥ 0.01 |
| Autocorrelation | Sample independence | \|r\| < 0.3 |

### Phase 3: Reservoir Computing Validation
| Test | What It Measures | Pass Criteria |
|------|------------------|---------------|
| Coefficient of Variation | Variance relative to mean | CV > 0.1 |
| Fading Memory | Autocorrelation decay | Decay rate > 50% |
| Separation Property | Input distinguishability | Variance ratio > 1.0 |
| Nonlinearity | Non-random dynamics | Z-score > 2.0 vs surrogates |

### Phase 4: RAG Utility Validation
| Test | What It Measures | Pass Criteria |
|------|------------------|---------------|
| Hash Collision Rate | Index uniqueness | Rate < 0.001 |
| LSH Correlation | Semantic similarity preservation | r ≥ 0.5 |
| Retrieval Precision | Hash-based retrieval quality | P@5 > 0.3 (better than random) |

## Expected Outcomes (Honest Prediction)

Based on the architecture, here's what the experiment will likely reveal:

### Likely to PASS ✅
- **Throughput QPS**: The hardware does produce ~0.18 events/sec
- **CV > 0.1**: Mining naturally shows variance (Poisson process)
- **Hash Collision Rate**: SHA256 is collision-resistant by design
- **Monobit/Runs Tests**: Mining timing should pass basic randomness

### Likely to FAIL ❌
- **Shannon Entropy ≥ 3.0**: The measured 0.0088 bits is FAR below cryptographic requirements
- **LSH Correlation ≥ 0.5**: SHA256 hashes are designed to NOT preserve similarity
- **Retrieval Precision > 0.3**: Hash-based retrieval cannot match semantic search
- **Nonlinearity (Reservoir)**: Mining is a Poisson process, not a dynamic reservoir

### Context-Dependent ⚠️
- **Fading Memory**: May pass by coincidence, not by design
- **Separation Property**: Depends on sample quality

## Key Insight

The fundamental issue is a **category error**: the paper conflates several distinct concepts:

1. **Poisson Process Statistics** → Confused with "Physical Chaos"
2. **SHA256 Cryptographic Hashing** → Confused with "Semantic Similarity"
3. **Mining Event Timing** → Confused with "Reservoir Dynamics"

A CV ≈ 1 for mining events is **exactly what probability theory predicts** for a Poisson process. This is not "exploitable chaos" - it's expected behavior.

## Output Files

- `chimera_validation_report.json`: Complete test results in JSON format
- Console output: Formatted report with PASS/FAIL verdicts

## Modifying Thresholds

Edit the `ExperimentConfig` class at the top of the script:

```python
@dataclass
class ExperimentConfig:
    # Adjust these based on your requirements
    ENTROPY_PASS_THRESHOLD: float = 3.0      # Lower = easier to pass
    CV_CHAOS_THRESHOLD: float = 0.1          # Lower = easier to pass
    LSH_SIMILARITY_CORRELATION_MIN: float = 0.5  # Lower = easier to pass
```

## Scientific Integrity

This experiment is designed to be **brutally honest**. It will:

1. Report actual measurements without interpretation bias
2. Use established statistical methods (NIST, Pearson, etc.)
3. Provide clear PASS/FAIL verdicts with thresholds
4. Distinguish between "technically true" and "scientifically meaningful"

The goal is to identify which claims are **defensible for peer review** and which need revision or removal.
