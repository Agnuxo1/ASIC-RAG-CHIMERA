from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DMDModel:
    A: np.ndarray  # (k,k)


@dataclass(frozen=True)
class LinearDecoder:
    W: np.ndarray  # (d,k)
    b: np.ndarray  # (d,)

    def decode(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        return (h @ self.W.T + self.b[None, :]).astype(np.float32)


def fit_dmd(h: np.ndarray, *, ridge: float = 1e-6) -> Tuple[DMDModel, Dict[str, Any]]:
    """
    Fit h_{t+1} ≈ A h_t using ridge-regularized least squares over all trajectories.
    h: (n_traj, steps, k) with steps>=2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError(f"expected h with shape (n,steps,k), got {h.shape}")
    n, steps, k = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    if steps < 2:
        raise ValueError("need steps>=2 for DMD fit")

    H0 = h[:, :-1, :].reshape(-1, k)
    H1 = h[:, 1:, :].reshape(-1, k)

    G = H0.T @ H0 + float(ridge) * np.eye(k, dtype=np.float64)
    B = H0.T @ H1
    A = np.linalg.solve(G, B).T  # (k,k)

    eig = np.linalg.eigvals(A.astype(np.complex128))
    spec = {
        "ridge": float(ridge),
        "n_pairs": int(H0.shape[0]),
        "k": int(k),
        "eig_abs_mean": float(np.mean(np.abs(eig))) if eig.size else 0.0,
        "eig_abs_max": float(np.max(np.abs(eig))) if eig.size else 0.0,
        "eig_neutral_count_abs_0p99_1p01": int(np.sum((np.abs(eig) >= 0.99) & (np.abs(eig) <= 1.01))),
    }
    return DMDModel(A=A.astype(np.float32)), spec


def fit_linear_decoder(h: np.ndarray, x: np.ndarray, *, ridge: float = 1e-6) -> Tuple[LinearDecoder, Dict[str, Any]]:
    """
    Fit x ≈ h W^T + b.
    h: (N,k), x: (N,d)
    """
    h = np.asarray(h, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if h.ndim != 2 or x.ndim != 2 or h.shape[0] != x.shape[0]:
        raise ValueError(f"bad shapes h={h.shape}, x={x.shape}")
    N, k = int(h.shape[0]), int(h.shape[1])
    d = int(x.shape[1])

    h_aug = np.concatenate([h, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,k+1)
    G = h_aug.T @ h_aug + float(ridge) * np.eye(k + 1, dtype=np.float64)
    B = h_aug.T @ x  # (k+1,d)
    theta = np.linalg.solve(G, B)  # (k+1,d)
    W = theta[:k, :].T  # (d,k)
    b = theta[k, :]  # (d,)
    return LinearDecoder(W=W.astype(np.float32), b=b.astype(np.float32)), {"ridge": float(ridge), "n": int(N), "k": int(k), "d": int(d)}


def predict_rollout(
    *,
    A: np.ndarray,
    decoder: LinearDecoder,
    h0: np.ndarray,
    horizon: int,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    h = np.asarray(h0, dtype=np.float64).reshape(1, -1)
    preds = []
    for _ in range(int(horizon)):
        h = (h @ A.T)
        preds.append(decoder.decode(h)[0])
    return np.asarray(preds, dtype=np.float32)

