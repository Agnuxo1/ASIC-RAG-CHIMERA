from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SurrogateSpec:
    kind: str  # none|shuffle|blockshuffle|phase|iaaft
    block_size: int = 16
    seed: int = 0
    n_iters: int = 20  # for iaaft


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def surrogate_shuffle(x: np.ndarray, *, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return x.astype(np.float32, copy=True)
    r = _rng(int(seed))
    idx = np.arange(int(x.size), dtype=np.int64)
    r.shuffle(idx)
    return x[idx].astype(np.float32, copy=False)


def surrogate_blockshuffle(x: np.ndarray, *, block_size: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return x.astype(np.float32, copy=True)
    b = int(max(1, block_size))
    n = int(x.size)
    m = int(((n + b - 1) // b) * b)
    y = x
    if m > n:
        y = np.pad(x, (0, m - n), mode="edge").astype(np.float32, copy=False)
    blocks = y.reshape(-1, b)
    r = _rng(int(seed))
    perm = np.arange(int(blocks.shape[0]), dtype=np.int64)
    r.shuffle(perm)
    z = blocks[perm, :].reshape(-1)[:n]
    return z.astype(np.float32, copy=False)


def surrogate_phase_randomize(x: np.ndarray, *, seed: int) -> np.ndarray:
    """
    Phase randomization surrogate: preserves FFT magnitude approximately, destroys phase structure.
    Deterministic given seed.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n <= 3:
        return x.astype(np.float32, copy=True)
    r = _rng(int(seed))
    x64 = x.astype(np.float64, copy=False)
    mu = float(np.mean(x64))
    sd = float(np.std(x64, ddof=1) if n >= 2 else 0.0)
    sd = float(sd if sd > 1e-9 else 1.0)
    y = x64 - mu

    Y = np.fft.rfft(y)
    mag = np.abs(Y)
    ang = np.angle(Y)

    # Randomize phases for bins 1..-2; keep DC and Nyquist (if present).
    new_ang = ang.copy()
    if new_ang.size > 2:
        new_ang[1:-1] = r.uniform(0.0, 2.0 * np.pi, size=int(new_ang.size - 2))
    elif new_ang.size == 2:
        pass

    Y2 = mag * np.exp(1j * new_ang)
    y2 = np.fft.irfft(Y2, n=n).real

    # Rescale to original mean/std for comparability.
    y2 = y2 - float(np.mean(y2))
    sd2 = float(np.std(y2, ddof=1) if n >= 2 else 0.0)
    sd2 = float(sd2 if sd2 > 1e-9 else 1.0)
    y2 = (y2 / sd2) * sd + mu
    return y2.astype(np.float32)


def surrogate_iaaft(x: np.ndarray, *, seed: int, n_iters: int = 20) -> np.ndarray:
    """
    Iterative Amplitude Adjusted Fourier Transform surrogate (1D).
    Approximate: matches target amplitude distribution and approximately matches FFT magnitudes.
    Deterministic given seed.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n <= 3:
        return x.astype(np.float32, copy=True)

    rng = _rng(int(seed))
    x64 = x.astype(np.float64, copy=False)

    # Target distribution via sorted values.
    x_sorted = np.sort(x64)

    # Target FFT magnitudes (of demeaned series).
    x0 = x64 - float(np.mean(x64))
    target_mag = np.abs(np.fft.rfft(x0))

    # Initialize with a random permutation of x.
    y = x64.copy()
    rng.shuffle(y)
    y = y - float(np.mean(y))

    n_iters = int(max(1, n_iters))
    for _ in range(n_iters):
        # Enforce spectrum magnitude.
        Y = np.fft.rfft(y)
        ang = np.angle(Y)
        Y2 = target_mag * np.exp(1j * ang)
        y_ifft = np.fft.irfft(Y2, n=n).real

        # Enforce amplitude distribution by rank-order mapping.
        order = np.argsort(y_ifft)
        y_new = np.empty_like(y_ifft)
        y_new[order] = x_sorted
        y = y_new - float(np.mean(y_new))

    # Final rescale to match mean/std of original x.
    mu = float(np.mean(x64))
    sd = float(np.std(x64, ddof=1) if n >= 2 else 0.0)
    sd = float(sd if sd > 1e-9 else 1.0)
    sd_y = float(np.std(y, ddof=1) if n >= 2 else 0.0)
    sd_y = float(sd_y if sd_y > 1e-9 else 1.0)
    y = (y / sd_y) * sd + mu
    return y.astype(np.float32)


def apply_surrogate(x: np.ndarray, spec: SurrogateSpec) -> Tuple[np.ndarray, Dict[str, object]]:
    kind = str(spec.kind).lower().strip()
    if kind in {"none", ""}:
        return np.asarray(x, dtype=np.float32).reshape(-1).astype(np.float32, copy=True), {"kind": "none"}
    if kind in {"shuffle", "permute"}:
        return surrogate_shuffle(x, seed=int(spec.seed)), {"kind": "shuffle", "seed": int(spec.seed)}
    if kind in {"blockshuffle", "block_shuffle"}:
        return (
            surrogate_blockshuffle(x, block_size=int(spec.block_size), seed=int(spec.seed)),
            {"kind": "blockshuffle", "seed": int(spec.seed), "block_size": int(spec.block_size)},
        )
    if kind in {"phase", "phase_randomize", "phase_randomized"}:
        return surrogate_phase_randomize(x, seed=int(spec.seed)), {"kind": "phase", "seed": int(spec.seed)}
    if kind in {"iaaft"}:
        n_iters = int(max(1, int(spec.n_iters)))
        y = surrogate_iaaft(x, seed=int(spec.seed), n_iters=n_iters)
        return y, {"kind": "iaaft", "seed": int(spec.seed), "n_iters": int(n_iters)}
    raise ValueError(f"unknown surrogate kind: {spec.kind!r}")

