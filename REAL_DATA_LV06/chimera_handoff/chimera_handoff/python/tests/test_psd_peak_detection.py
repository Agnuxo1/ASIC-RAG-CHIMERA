from __future__ import annotations

import numpy as np

from chimera_handoff.entropy.chronos_metrics import psd_peak_features


def _sine_counts(*, f_hz: float, dt_s: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(n), dtype=np.float64) * float(dt_s)
    base = 20.0 + 8.0 * np.sin(2.0 * np.pi * float(f_hz) * t)
    noise = rng.normal(0.0, 0.8, size=int(n))
    x = np.maximum(0.0, base + noise)
    return x.astype(np.float64)


def test_psd_peak_detects_heartbeat_2p4() -> None:
    dt = 0.05
    counts = _sine_counts(f_hz=2.4, dt_s=dt, n=2048, seed=1)
    psd = psd_peak_features(counts, bin_dt_s=dt, f_min_hz=0.5, f_max_hz=10.0)
    assert abs(float(psd["psd_peak_hz"]) - 2.4) < 0.1
    assert float(psd["psd_peak_snr_db"]) > 6.0
    assert float(psd["psd_peak_q"]) > 1.0


def test_psd_peak_detects_heartbeat_3p7() -> None:
    dt = 0.05
    counts = _sine_counts(f_hz=3.7, dt_s=dt, n=2048, seed=2)
    psd = psd_peak_features(counts, bin_dt_s=dt, f_min_hz=0.5, f_max_hz=10.0)
    assert abs(float(psd["psd_peak_hz"]) - 3.7) < 0.1
    assert float(psd["psd_peak_snr_db"]) > 6.0

