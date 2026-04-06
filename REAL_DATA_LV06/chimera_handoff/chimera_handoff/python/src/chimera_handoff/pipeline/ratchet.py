from __future__ import annotations

import numpy as np


def apply_J(u: np.ndarray, *, top: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    return np.clip(np.maximum(u, 0.0), 0.0, float(top)).astype(np.float32, copy=False)


def choose_top(u: np.ndarray, *, quantile: float, min_top: float) -> float:
    u = np.asarray(u, dtype=np.float32)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float(min_top)
    q = float(np.quantile(u.reshape(-1), float(quantile)))
    if not np.isfinite(q):
        return float(min_top)
    return float(max(float(min_top), q))


def sign_split_relu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pos = np.maximum(x, 0.0)
    neg = np.maximum(-x, 0.0)
    return np.concatenate([pos, neg], axis=-1).astype(np.float32, copy=False)

