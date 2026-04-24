from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RClosureConfig:
    mode: str = "off"  # off|diag|apply
    q: float = 0.8
    min_support: int = 20
    p_thr: float = 0.9
    max_iters: int = 32
    laplace_alpha: float = 1.0


def _as_bool(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def binarize_by_feature_quantile(u: np.ndarray, *, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    u: (n_windows, k) float features
    returns:
      active: (n_windows, k) bool
      thr: (k,) float thresholds
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("u must be (n_windows, k)")
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    thr = np.quantile(u, q, axis=0).astype(np.float64)
    active = (u >= thr[None, :]) & np.isfinite(u)
    return _as_bool(active), thr.astype(np.float64)


def learn_implication_graph(
    active: np.ndarray,
    *,
    min_support: int,
    p_thr: float,
    laplace_alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    active: (n_windows, k) bool, from calibration windows.
    Learn edges i->j when P(j|i) >= p_thr with support(i) >= min_support.
    """
    A = _as_bool(active)
    if A.ndim != 2:
        raise ValueError("active must be (n_windows, k)")
    n, k = int(A.shape[0]), int(A.shape[1])
    min_support = int(max(1, min_support))
    p_thr = float(p_thr)
    alpha = float(max(0.0, laplace_alpha))

    support = A.sum(axis=0).astype(np.int64)  # (k,)
    # Co-occurrence counts: C[i,j] = count(i active and j active).
    C = (A.astype(np.int64).T @ A.astype(np.int64)).astype(np.int64)  # (k,k)

    edges: List[Dict[str, Any]] = []
    adj = np.zeros((k, k), dtype=bool)
    for i in range(k):
        si = int(support[i])
        if si < min_support:
            continue
        denom = float(si + 2.0 * alpha) if alpha > 0 else float(si)
        if denom <= 0:
            continue
        for j in range(k):
            if i == j:
                continue
            cij = int(C[i, j])
            num = float(cij + alpha) if alpha > 0 else float(cij)
            p = float(num / denom)
            if p >= p_thr:
                adj[i, j] = True
                edges.append({"i": int(i), "j": int(j), "p_j_given_i": float(p), "support_i": int(si), "count_ij": int(cij)})

    mutual_pairs = []
    mutual_edge_count = 0
    for i in range(k):
        for j in range(i + 1, k):
            if bool(adj[i, j]) and bool(adj[j, i]):
                mutual_edge_count += 1
                mutual_pairs.append({"i": int(i), "j": int(j)})

    density = float(adj.sum() / max(1, k * (k - 1)))

    return {
        "n_windows": int(n),
        "k": int(k),
        "min_support": int(min_support),
        "p_thr": float(p_thr),
        "laplace_alpha": float(alpha),
        "support": support.astype(int).tolist(),
        "edges": edges,
        "graph_density": float(density),
        "mutual_edge_count": int(mutual_edge_count),
        "mutual_pairs": mutual_pairs,
        # For efficient downstream closure.
        "adj_bool": adj,
    }


def closure_iterate(
    s0: np.ndarray,
    *,
    adj: np.ndarray,
    max_iters: int,
) -> Tuple[np.ndarray, int]:
    """
    s0: (k,) bool
    adj: (k,k) bool adjacency (i->j)
    returns:
      s_final: (k,) bool
      birth: number of iterations to reach fixed point (0 if already fixed)
    """
    s = _as_bool(s0).reshape(-1)
    adj = _as_bool(adj)
    k = int(s.size)
    if adj.shape != (k, k):
        raise ValueError("adj shape mismatch")
    max_iters = int(max(0, max_iters))
    birth = 0
    for t in range(max_iters):
        nxt = s | (s @ adj)
        if np.array_equal(nxt, s):
            break
        s = nxt
        birth += 1
    return _as_bool(s), int(birth)


def apply_r_closure_diag(
    u: np.ndarray,
    *,
    thr: np.ndarray,
    adj: np.ndarray,
    mutual_pairs: List[Dict[str, Any]],
    max_iters: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute per-window closure stats on u without modifying it.
    Returns (summary_stats, timeseries_rows).
    """
    u = np.asarray(u, dtype=np.float64)
    thr = np.asarray(thr, dtype=np.float64).reshape(-1)
    n, k = int(u.shape[0]), int(u.shape[1])
    if thr.size != k:
        raise ValueError("thr size mismatch")

    births = np.zeros((n,), dtype=np.int64)
    added = np.zeros((n,), dtype=np.int64)
    mutual_active = np.zeros((n,), dtype=np.int64)

    for t in range(n):
        s0 = (u[t] >= thr) & np.isfinite(u[t])
        s_final, b = closure_iterate(s0, adj=adj, max_iters=int(max_iters))
        births[t] = int(b)
        added[t] = int(np.sum(s_final) - np.sum(s0))
        # Mutual-active pairs count.
        c = 0
        for p in mutual_pairs:
            i = int(p["i"])
            j = int(p["j"])
            if bool(s_final[i]) and bool(s_final[j]):
                c += 1
        mutual_active[t] = int(c)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(np.asarray(x, dtype=np.float64), float(p))) if x.size else 0.0

    stats = {
        "birth_mean": float(births.mean()) if births.size else 0.0,
        "birth_p50": q(births, 0.50),
        "birth_p90": q(births, 0.90),
        "birth_p95": q(births, 0.95),
        "birth_max": int(births.max()) if births.size else 0,
        "closure_added_mean": float(added.mean()) if added.size else 0.0,
        "mutual_active_pairs_rate_mean": float(mutual_active.mean()) if mutual_active.size else 0.0,
        "mutual_active_pairs_rate_p90": q(mutual_active, 0.90),
        "n_windows": int(n),
    }
    rows = [
        {"window_idx": int(i), "birth_R": int(births[i]), "added_count": int(added[i]), "mutual_active_pairs": int(mutual_active[i])}
        for i in range(n)
    ]
    return stats, rows


def apply_r_closure_to_u(
    u: np.ndarray,
    *,
    thr: np.ndarray,
    adj: np.ndarray,
    max_iters: int,
) -> np.ndarray:
    """
    Apply closure by forcing implied activations to at least thr[j] (preserves magnitudes).
    """
    u = np.asarray(u, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32).reshape(-1)
    n, k = int(u.shape[0]), int(u.shape[1])
    if thr.size != k:
        raise ValueError("thr size mismatch")
    out = u.copy()
    for t in range(n):
        s0 = (out[t].astype(np.float64) >= thr.astype(np.float64)) & np.isfinite(out[t].astype(np.float64))
        s_final, _b = closure_iterate(s0, adj=adj, max_iters=int(max_iters))
        # Activate implied features to at least threshold.
        mask = s_final & (~s0)
        if np.any(mask):
            out[t, mask] = np.maximum(out[t, mask], thr[mask])
    return out


def write_implication_graph(path: Path, graph: Dict[str, Any], *, thr: np.ndarray, q: float) -> None:
    obj = {k: v for k, v in graph.items() if k != "adj_bool"}
    obj["thresholds"] = np.asarray(thr, dtype=np.float64).tolist()
    obj["threshold_quantile_q"] = float(q)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_implication_graph(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load a graph written by write_implication_graph and reconstruct adjacency.
    Returns (adj_bool, thresholds, mutual_pairs, raw_json).
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    k = int(obj.get("k", 0))
    if k <= 0:
        raise ValueError("invalid implication graph: missing k")
    thr = np.asarray(obj.get("thresholds", []), dtype=np.float64).reshape(-1)
    if thr.size != k:
        raise ValueError("invalid implication graph: thresholds size mismatch")
    edges = list(obj.get("edges", []))
    adj = np.zeros((k, k), dtype=bool)
    for e in edges:
        i = int(e.get("i"))
        j = int(e.get("j"))
        if 0 <= i < k and 0 <= j < k and i != j:
            adj[i, j] = True
    mutual_pairs = list(obj.get("mutual_pairs", []))
    return adj, thr, mutual_pairs, obj


def write_r_closure_timeseries_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window_idx", "birth_R", "added_count", "mutual_active_pairs"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
