from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from chimera_handoff.util.paths import ensure_ml_on_path


def heyting_metrics_from_h(h: np.ndarray, *, top: float, mutual_thr: float = 0.95, adj_n: int = 4096, seed: int = 0) -> Dict[str, Any]:
    """
    h: (n_traj, steps, k) in [0,top]^k
    Returns a jsonable dict with key Heyting sanity/stability metrics.
    """
    ensure_ml_on_path()
    try:
        import torch
        from eigenlearner.heyting_ops import BoundedHeyting, check_himp_adjoint  # type: ignore
    except Exception as e:
        h = np.asarray(h, dtype=np.float32)
        n, steps, k = (int(h.shape[0]), int(h.shape[1]), int(h.shape[2])) if h.ndim == 3 else (0, 0, 0)
        return {
            "error": "missing_optional_deps(torch,eigenlearner)",
            "error_detail": str(e),
            "top": float(top),
            "n_traj": int(n),
            "steps": int(steps),
            "k": int(k),
            "boundary_max_abs": 0.0,
            "regularity_gap": {"mean_l1": 0.0, "max_linf": 0.0},
            "double_neg_change_frac": 0.0,
            "essence": {"mean_std": 0.0},
            "envelope": {"mean_std": 0.0},
            "implication": {"mutual_thr": float(mutual_thr), "mutual_pairs_count": 0, "mutual_mean_offdiag": 0.0},
            "anomaly": {"split": 0, "score_l1_mean": 0.0, "score_l1_std": 0.0, "score_l1_max": 0.0},
            "adjunction": {"n": int(adj_n), "mismatches": 0},
        }

    h = np.asarray(h, dtype=np.float32)
    n, steps, k = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    z = torch.from_numpy(h).to(dtype=torch.float32)
    hey = BoundedHeyting(top=float(top))
    z = hey.project(z)

    # Per-trajectory essence/envelope.
    ess = z.amin(dim=1)  # (n,k)
    env = z.amax(dim=1)  # (n,k)

    # Implication matrix between essences.
    impl = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ai = ess[i][None, :].expand(n, -1)
        impl[i, :] = hey.leq_frac(ai, ess).detach().cpu().numpy().astype(np.float64, copy=False)
    mutual = 0.5 * (impl + impl.T)
    mutual_pairs = int(np.sum(np.triu(mutual >= float(mutual_thr), k=1)))

    # Flatten all timepoints for "stream" stats.
    flat = z.reshape(-1, k)
    stats = hey.stats(flat)
    gap = hey.regularity_gap(flat)
    boundary = hey.meet(flat, hey.hnot(flat))

    # Anomaly score: safe envelope from first half of trajectories, measure violations on second half.
    split = n // 2
    normal_env = env[: max(1, split)].amax(dim=0)
    test_z = z[split:] if split < n else z[:0]
    viol = torch.clamp(test_z - normal_env[None, None, :], min=0.0)
    score_l1 = viol.sum(dim=-1).mean(dim=-1) if viol.numel() else torch.zeros((0,), dtype=torch.float32)

    # Adjunction mismatches on random triples of timepoints from the flattened stream.
    rng = np.random.default_rng(int(seed))
    if flat.shape[0] > 0:
        idx = rng.integers(0, int(flat.shape[0]), size=int(adj_n), endpoint=False)
        jdx = rng.integers(0, int(flat.shape[0]), size=int(adj_n), endpoint=False)
        kdx = rng.integers(0, int(flat.shape[0]), size=int(adj_n), endpoint=False)
        a = flat[torch.from_numpy(idx)]
        b = flat[torch.from_numpy(jdx)]
        c = flat[torch.from_numpy(kdx)]
        ok = check_himp_adjoint(hey, a, b, c)
        mism = int((~ok).sum().item())
    else:
        mism = 0

    return {
        "top": float(top),
        "n_traj": int(n),
        "steps": int(steps),
        "k": int(k),
        "boundary_max_abs": float(boundary.abs().amax().item()) if boundary.numel() else 0.0,
        "regularity_gap": {
            "mean_l1": float(gap.abs().sum(dim=-1).mean().item()) if gap.numel() else 0.0,
            "max_linf": float(gap.abs().amax().item()) if gap.numel() else 0.0,
        },
        "double_neg_change_frac": float(stats.double_neg_change_frac),
        "essence": {"mean_std": float(ess.std(dim=0).mean().item()) if ess.numel() else 0.0},
        "envelope": {"mean_std": float(env.std(dim=0).mean().item()) if env.numel() else 0.0},
        "implication": {
            "mutual_thr": float(mutual_thr),
            "mutual_pairs_count": int(mutual_pairs),
            "mutual_mean_offdiag": float((mutual.sum() - np.trace(mutual)) / max(1, n * (n - 1))),
        },
        "anomaly": {
            "split": int(split),
            "score_l1_mean": float(score_l1.mean().item()) if score_l1.numel() else 0.0,
            "score_l1_std": float(score_l1.std().item()) if score_l1.numel() else 0.0,
            "score_l1_max": float(score_l1.max().item()) if score_l1.numel() else 0.0,
        },
        "adjunction": {"n": int(adj_n), "mismatches": int(mism)},
    }
