from __future__ import annotations

import numpy as np

from chimera_handoff.entropy.chronos_metrics import psd_peak_features


def test_batching_peak_at_inverse_flush_period() -> None:
    # A flush period of 200ms implies an expected peak around 5Hz.
    dt = 0.05
    f_expected = 5.0
    n = 2048
    t = np.arange(int(n), dtype=np.float64) * dt
    # Use a sinusoid to avoid harmonics dominating the peak picker.
    counts = 20.0 + 8.0 * np.sin(2.0 * np.pi * f_expected * t)
    counts += np.random.default_rng(0).normal(0.0, 0.6, size=int(n))
    counts = np.maximum(0.0, counts)
    psd = psd_peak_features(counts, bin_dt_s=dt, f_min_hz=0.5, f_max_hz=10.0)
    assert abs(float(psd["psd_peak_hz"]) - f_expected) < 0.2
