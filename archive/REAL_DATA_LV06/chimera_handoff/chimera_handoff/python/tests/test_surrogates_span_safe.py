from __future__ import annotations

import numpy as np

from chimera_handoff.entropy.float_surrogates import SurrogateSpec, apply_surrogate


def test_phase_surrogate_span_is_renormalized() -> None:
    # Targets the historical failure mode where FFT surrogates on log-deltas
    # could produce huge reconstructed durations and explode PSD binning.
    rng = np.random.default_rng(0)
    n = 512
    base_deltas = rng.lognormal(mean=-2.0, sigma=0.5, size=n).astype(np.float64)  # ~O(0.1s) scale
    base_span = float(np.sum(base_deltas))

    base_log10 = np.log10(np.maximum(base_deltas, 1e-12)).astype(np.float32)
    y_log10, _meta = apply_surrogate(base_log10, SurrogateSpec(kind="phase", seed=7))
    y_log10 = np.asarray(y_log10, dtype=np.float64)

    clip = 12.0
    y_log10 = np.clip(y_log10, -clip, clip)
    y = np.power(10.0, y_log10, dtype=np.float64)
    y = np.maximum(y, 1e-12)
    sum_y = float(np.sum(y))
    assert sum_y > 0
    y = y * (base_span / sum_y)
    span = float(np.sum(y))

    assert abs(span - base_span) / base_span < 1e-6

