from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def hist16(x: np.ndarray, *, clip_k: float = 5.0) -> np.ndarray:
    """
    16-bin histogram over [-clip_k, clip_k], normalized to sum to 1.
    """
    x = _as_f64(x)
    if x.size == 0:
        return np.zeros((16,), dtype=np.float32)
    ck = float(clip_k)
    if not (ck > 0):
        ck = 5.0
    y = np.clip(x, -ck, ck)
    h, _edges = np.histogram(y, bins=16, range=(-ck, ck), density=False)
    h = h.astype(np.float64)
    h = h / max(1.0, float(h.sum()))
    return h.astype(np.float32, copy=False)


def signcount16(x: np.ndarray) -> np.ndarray:
    """
    16 features: fraction of positives in each of 16 equal-length blocks.
    """
    x = _as_f64(x)
    if x.size == 0:
        return np.zeros((16,), dtype=np.float32)
    n = int(x.size)
    # Pad to a multiple of 16.
    m = int(((n + 15) // 16) * 16)
    if m > n:
        x = np.pad(x, (0, m - n), mode="edge")
    blocks = x.reshape(16, -1)
    frac_pos = (blocks > 0).mean(axis=1)
    return frac_pos.astype(np.float32, copy=False)


def fft8(x: np.ndarray) -> np.ndarray:
    """
    8 FFT magnitude features: first 8 non-DC rFFT bins (padded/truncated as needed).
    """
    x = _as_f64(x)
    if x.size == 0:
        return np.zeros((8,), dtype=np.float32)
    y = x - float(np.mean(x))
    z = np.fft.rfft(y)
    mag = np.abs(z).astype(np.float64)
    mag = mag[1:9]  # drop DC, keep up to 8 bins
    if mag.size < 8:
        mag = np.pad(mag, (0, 8 - int(mag.size)), mode="constant")
    # Scale-invariant-ish: log1p.
    out = np.log1p(mag)
    return out.astype(np.float32, copy=False)


def runs8(x: np.ndarray) -> np.ndarray:
    """
    8 run/structure statistics of sign(x):
      [mean_run, std_run, max_run, n_runs, frac_pos, frac_zero, frac_neg, transition_rate]
    """
    x = _as_f64(x)
    if x.size == 0:
        return np.zeros((8,), dtype=np.float32)
    s = np.zeros((x.size,), dtype=np.int8)
    s[x > 0] = 1
    s[x < 0] = -1

    runs = []
    run = 1
    transitions = 0
    for i in range(1, int(s.size)):
        if int(s[i]) == int(s[i - 1]):
            run += 1
        else:
            transitions += 1
            runs.append(run)
            run = 1
    runs.append(run)
    r = np.asarray(runs, dtype=np.float64)

    mean_run = float(r.mean()) if r.size else 0.0
    std_run = float(r.std(ddof=1)) if r.size >= 2 else 0.0
    max_run = float(r.max()) if r.size else 0.0
    n_runs = float(r.size)
    frac_pos = float(np.mean(s == 1))
    frac_zero = float(np.mean(s == 0))
    frac_neg = float(np.mean(s == -1))
    transition_rate = float(transitions / max(1, int(s.size - 1)))
    out = np.asarray([mean_run, std_run, max_run, n_runs, frac_pos, frac_zero, frac_neg, transition_rate], dtype=np.float32)
    return out


def float_context_features(x: np.ndarray, *, clip_k: float = 5.0) -> np.ndarray:
    """
    Concatenated feature vector: hist16 || signcount16 || fft8 || runs8 (48 dims).
    """
    h = hist16(x, clip_k=float(clip_k))
    s = signcount16(x)
    f = fft8(x)
    r = runs8(x)
    return np.concatenate([h, s, f, r], axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class FloatContextSummary:
    dim: int
    parts: Dict[str, Tuple[int, int]]


def float_context_summary() -> FloatContextSummary:
    # Static summary for downstream wiring.
    return FloatContextSummary(
        dim=48,
        parts={"hist16": (0, 16), "signcount16": (16, 32), "fft8": (32, 40), "runs8": (40, 48)},
    )

