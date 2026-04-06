from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SpectralSpec:
    window_L: int = 64
    signed_log1p: bool = True
    standardize: bool = True


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def _windows_2d(x: np.ndarray, *, L: int) -> np.ndarray:
    """
    x: (T,d)
    returns: (T, L, d) windows ending at each t, left-padded.
    """
    x = np.asarray(x, dtype=np.float32)
    T, d = int(x.shape[0]), int(x.shape[1])
    if T <= 0:
        return np.zeros((0, L, d), dtype=np.float32)
    out = np.empty((T, L, d), dtype=np.float32)
    for t in range(T):
        # Window ending at t: indices [t-L+1, ..., t], pad on the left if needed.
        start = t - L + 1
        if start >= 0:
            out[t] = x[start : t + 1, :]
        else:
            pad_n = -start
            # Pad with x[0] to avoid introducing artificial zeros (scale-preserving).
            out[t, :pad_n, :] = x[0:1, :].repeat(pad_n, axis=0)
            out[t, pad_n:, :] = x[0 : t + 1, :]
    return out


def rfft_embed_windowed(traj: np.ndarray, *, spec: SpectralSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    traj: (n_traj, T, d)
    returns:
      c: (n_traj, T, k) signed features (pre sign-split/J), using left-padded windows
      meta
    """
    traj = np.asarray(traj, dtype=np.float32)
    n, T, d = int(traj.shape[0]), int(traj.shape[1]), int(traj.shape[2])
    L = int(spec.window_L)
    if T < L:
        raise ValueError(f"time_steps={T} < window_L={L}")
    f_bins = L // 2 + 1
    k = d * f_bins * 2  # real+imag

    out = np.empty((n, T, k), dtype=np.float32)
    for i in range(n):
        w = _windows_2d(traj[i], L=L)  # (T, L, d)
        # rfft along the time axis for each window and each dim.
        # We'll pack [Re, Im] for each dim.
        feats = []
        for j in range(d):
            y = np.fft.rfft(w[:, :, j], axis=1)  # (n_win, f_bins)
            feats.append(y.real.astype(np.float32))
            feats.append(y.imag.astype(np.float32))
        c = np.concatenate(feats, axis=1)  # (n_win, k)
        if spec.signed_log1p:
            c = _signed_log1p(c)
        out[i] = c

    meta: Dict[str, Any] = {
        "kind": "rfft",
        "window_L": int(L),
        "f_bins": int(f_bins),
        "signed_log1p": bool(spec.signed_log1p),
        "standardize": bool(spec.standardize),
        "k_raw": int(k),
        "padding": {"mode": "repeat_first"},
    }
    return out, meta


def standardize_train_apply(train_c: np.ndarray, test_c: np.ndarray, *, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train_c = np.asarray(train_c, dtype=np.float32)
    test_c = np.asarray(test_c, dtype=np.float32)
    flat = train_c.reshape(-1, train_c.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0, ddof=1)
    std = np.where(std > eps, std, np.float32(1.0))
    train_z = (train_c - mean[None, None, :]) / std[None, None, :]
    test_z = (test_c - mean[None, None, :]) / std[None, None, :]
    return train_z.astype(np.float32, copy=False), test_z.astype(np.float32, copy=False), {"mean": mean.tolist(), "std": std.tolist()}
