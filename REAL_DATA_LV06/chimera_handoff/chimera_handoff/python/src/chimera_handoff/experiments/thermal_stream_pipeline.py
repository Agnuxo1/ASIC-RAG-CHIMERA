from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.health import run_health_tests
from chimera_handoff.pipeline.dmd import fit_dmd
from chimera_handoff.pipeline.heyting import heyting_metrics_from_h
from chimera_handoff.pipeline.ratchet import apply_J, choose_top
from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats_arr(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1) if x.size >= 2 else 0.0),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _sat_fracs(h: np.ndarray, *, top: float, eps: float = 1e-6) -> Dict[str, float]:
    h = np.asarray(h, dtype=np.float64)
    if h.size == 0:
        return {"frac_at_0": 0.0, "frac_at_top": 0.0}
    frac0 = float(np.mean(h <= eps))
    fractop = float(np.mean(h >= float(top) - eps))
    return {"frac_at_0": frac0, "frac_at_top": fractop}


def _eig_summaries(A: np.ndarray) -> Dict[str, Any]:
    A = np.asarray(A, dtype=np.float64)
    eig = np.linalg.eigvals(A.astype(np.complex128))
    abs_e = np.abs(eig)
    ang = np.angle(eig)
    sr = float(np.max(abs_e)) if abs_e.size else 0.0
    neutral_0p98_1p02 = int(np.sum((abs_e >= 0.98) & (abs_e <= 1.02)))
    neutral_0p95_1p05 = int(np.sum((abs_e >= 0.95) & (abs_e <= 1.05)))
    band = (abs_e >= 0.98) & (abs_e <= 1.02)
    ang_band = np.abs(ang[band]) if np.any(band) else np.zeros((0,), dtype=np.float64)
    return {
        "eig": [{"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)), "angle": float(np.angle(z))} for z in eig],
        "spectral_radius": float(sr),
        "eig_abs_mean": float(np.mean(abs_e)) if abs_e.size else 0.0,
        "eig_abs_max": float(np.max(abs_e)) if abs_e.size else 0.0,
        "neutral_count_abs_0p98_1p02": int(neutral_0p98_1p02),
        "neutral_count_abs_0p95_1p05": int(neutral_0p95_1p05),
        "neutral_angle_abs_mean_in_0p98_1p02": float(np.mean(ang_band)) if ang_band.size else 0.0,
        "neutral_angle_abs_p95_in_0p98_1p02": float(np.quantile(ang_band, 0.95)) if ang_band.size else 0.0,
    }


def _bootstrap_dmd_spectrum(
    h_train: np.ndarray,
    *,
    ridge: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    h_train = np.asarray(h_train, dtype=np.float64)
    n, steps, k = h_train.shape
    if n <= 1 or n_boot <= 0 or steps < 2:
        return {"n_boot": int(n_boot), "error": "insufficient_data"}

    rng = np.random.default_rng(int(seed))
    n_boot = int(min(int(n_boot), 50))
    H0_all = h_train[:, :-1, :].reshape(-1, k)
    H1_all = h_train[:, 1:, :].reshape(-1, k)
    n_pairs = int(H0_all.shape[0])
    n_pairs_boot = int(min(4096, n_pairs))

    sr = np.empty((int(n_boot),), dtype=np.float64)
    n98 = np.empty((int(n_boot),), dtype=np.float64)
    eye = np.eye(int(k), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n_pairs, size=n_pairs_boot, endpoint=False)
        H0 = H0_all[idx, :]
        H1 = H1_all[idx, :]
        G = H0.T @ H0 + float(ridge) * eye
        B = H0.T @ H1
        A = np.linalg.solve(G, B).T.astype(np.float32)
        summ = _eig_summaries(A)
        sr[i] = float(summ["spectral_radius"])
        n98[i] = float(summ["neutral_count_abs_0p98_1p02"])

    def _ci(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1) if x.size >= 2 else 0.0),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }

    return {
        "n_boot": int(n_boot),
        "n_pairs_total": int(n_pairs),
        "n_pairs_boot": int(n_pairs_boot),
        "spectral_radius": _ci(sr),
        "neutral_count_abs_0p98_1p02": _ci(n98),
    }


def _windows_1d(x: np.ndarray, *, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = int(x.size)
    if T <= 0:
        return np.zeros((0, L), dtype=np.float32)
    out = np.empty((T, L), dtype=np.float32)
    for t in range(T):
        start = t - L + 1
        if start >= 0:
            out[t] = x[start : t + 1]
        else:
            pad_n = -start
            out[t, :pad_n] = float(x[0])
            out[t, pad_n:] = x[0 : t + 1]
    return out


def build_context_features(traj: np.ndarray, *, window_L: int, clip_k: float) -> np.ndarray:
    """
    traj: (n_traj, T) float time series, already normalized/clipped as desired.
    returns u: (n_traj, T, k) nonnegative context features.
    """
    from chimera_handoff.entropy.float_contexts import float_context_features

    traj = np.asarray(traj, dtype=np.float32)
    n, T = int(traj.shape[0]), int(traj.shape[1])
    L = int(window_L)
    if T < L:
        raise ValueError(f"time_steps={T} < window_L={L}")
    # Features are 48-d.
    k = int(float_context_features(np.zeros((L,), dtype=np.float32)).shape[0])
    out = np.empty((n, T, k), dtype=np.float32)
    for i in range(n):
        w = _windows_1d(traj[i], L=L)  # (T, L)
        for t in range(T):
            out[i, t, :] = float_context_features(w[t], clip_k=float(clip_k))
    return out


def _split_by_trajectory(traj: np.ndarray, *, train_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(traj.shape[0])
    n_train = max(1, int(round(float(train_frac) * n)))
    return traj[:n_train], traj[n_train:]


def _project_h(h: np.ndarray, *, top: float) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return np.clip(np.maximum(h, 0.0), 0.0, float(top))


def _predict_rollout_h(A: np.ndarray, h0: np.ndarray, *, top: float, horizon: int) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    h = _project_h(np.asarray(h0, dtype=np.float64).reshape(-1), top=float(top))
    out = np.empty((int(horizon), int(h.size)), dtype=np.float64)
    for t in range(int(horizon)):
        h = _project_h(h @ A.T, top=float(top))
        out[t] = h
    return out.astype(np.float32)


@dataclass(frozen=True)
class ThermalStreamConfig:
    out_dir: Path
    seed: int = 0
    source_id: str = "prng"
    sigma: float = 0.0  # kept for summarizer compatibility; unused

    n_traj: int = 2
    time_steps: int = 96
    train_frac: float = 0.5
    window_L: int = 64
    clip_k: float = 5.0

    rollout_horizon: int = 25
    ridge_dmd: float = 1e-6

    # If set, use a fixed top for J across sources (comparability governance).
    top_override: Optional[float] = None

    # Optional iterated closure (R̂) on binarized features.
    r_closure: str = "off"  # off|diag|apply
    r_q: float = 0.8
    r_min_support: int = 20
    r_p_thr: float = 0.9
    r_max_iters: int = 32

    mutual_thr: float = 0.95
    adj_n: int = 4096


def run_thermal_stream_from_series(cfg: ThermalStreamConfig, *, series: np.ndarray, series_meta: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = ensure_out_dir(Path(cfg.out_dir))

    x = np.asarray(series, dtype=np.float32).reshape(-1)
    if x.size < int(cfg.n_traj) * int(cfg.time_steps):
        raise ValueError("series too short for requested n_traj*time_steps")

    # Health tests on the float stream itself.
    health = run_health_tests(x)
    _write_json(out_dir / "entropy_health.json", {"source_id": str(cfg.source_id), "stream_kind": "float", **health.__dict__})

    traj = x[: int(cfg.n_traj) * int(cfg.time_steps)].reshape(int(cfg.n_traj), int(cfg.time_steps))
    train_traj, test_traj = _split_by_trajectory(traj, train_frac=float(cfg.train_frac))

    u_train = build_context_features(train_traj, window_L=int(cfg.window_L), clip_k=float(cfg.clip_k))
    u_test = build_context_features(test_traj, window_L=int(cfg.window_L), clip_k=float(cfg.clip_k))

    r_mode = str(cfg.r_closure).lower().strip()
    r_summary: Optional[Dict[str, Any]] = None
    if r_mode in {"diag", "apply"}:
        from chimera_handoff.pipeline.r_closure import (
            apply_r_closure_diag,
            apply_r_closure_to_u,
            binarize_by_feature_quantile,
            learn_implication_graph,
            write_implication_graph,
            write_r_closure_timeseries_csv,
        )

        # Calibration windows from train only.
        u_cal = u_train.reshape(-1, u_train.shape[-1]).astype(np.float64)
        active, thr = binarize_by_feature_quantile(u_cal, q=float(cfg.r_q))
        graph = learn_implication_graph(active, min_support=int(cfg.r_min_support), p_thr=float(cfg.r_p_thr), laplace_alpha=1.0)
        adj = np.asarray(graph["adj_bool"], dtype=bool)
        mutual_pairs = list(graph.get("mutual_pairs", []))

        write_implication_graph(out_dir / "implication_graph.json", graph, thr=thr, q=float(cfg.r_q))

        # Diagnostics on test windows (flattened).
        u_test_flat = u_test.reshape(-1, u_test.shape[-1]).astype(np.float64)
        stats, rows = apply_r_closure_diag(u_test_flat, thr=thr, adj=adj, mutual_pairs=mutual_pairs, max_iters=int(cfg.r_max_iters))
        r_summary = {
            "mode": str(r_mode),
            "q": float(cfg.r_q),
            "min_support": int(cfg.r_min_support),
            "p_thr": float(cfg.r_p_thr),
            "max_iters": int(cfg.r_max_iters),
            "graph_density": float(graph.get("graph_density", 0.0)),
            "mutual_edge_count": int(graph.get("mutual_edge_count", 0)),
            **stats,
        }
        _write_json(out_dir / "r_closure_stats.json", r_summary)
        write_r_closure_timeseries_csv(out_dir / "r_closure_timeseries.csv", rows)

        if r_mode == "apply":
            u_train = apply_r_closure_to_u(u_train.reshape(-1, u_train.shape[-1]), thr=thr, adj=adj, max_iters=int(cfg.r_max_iters)).reshape(u_train.shape)
            u_test = apply_r_closure_to_u(u_test.reshape(-1, u_test.shape[-1]), thr=thr, adj=adj, max_iters=int(cfg.r_max_iters)).reshape(u_test.shape)

    top = float(cfg.top_override) if cfg.top_override is not None else choose_top(u_train, quantile=0.999, min_top=1.0)
    h_train = apply_J(u_train, top=float(top))
    h_test = apply_J(u_test, top=float(top))

    dmd, dmd_meta = fit_dmd(h_train, ridge=float(cfg.ridge_dmd))
    A = np.asarray(dmd.A, dtype=np.float32)
    spec = _eig_summaries(A)
    boot = _bootstrap_dmd_spectrum(h_train, ridge=float(cfg.ridge_dmd), n_boot=50, seed=int(cfg.seed) + 1007)

    # One-step prediction error in h-space.
    if h_test.shape[0] >= 1 and h_test.shape[1] >= 2:
        h0 = h_test[:, :-1, :].reshape(-1, h_test.shape[-1]).astype(np.float64)
        h1_true = h_test[:, 1:, :].reshape(-1, h_test.shape[-1]).astype(np.float64)
        h1_hat = _project_h(h0 @ A.T.astype(np.float64), top=float(top)).astype(np.float64)
        pred_mse = float(np.mean((h1_hat - h1_true) ** 2)) if h1_true.size else 0.0
    else:
        pred_mse = 0.0

    # Rollout error in h-space (mean over horizon and dims).
    H = int(min(int(cfg.rollout_horizon), max(0, int(h_test.shape[1] - 1))))
    mse_by_t = []
    for i in range(int(h_test.shape[0])):
        if H <= 0:
            continue
        h_init = h_test[i, 0, :]
        h_true = h_test[i, 1 : 1 + H, :]
        h_hat = _predict_rollout_h(A=A, h0=h_init, top=float(top), horizon=H)
        if h_hat.shape != h_true.shape:
            continue
        mse_by_t.append(np.mean((h_hat - h_true) ** 2, axis=1))
    mse_by_t = np.stack(mse_by_t, axis=0) if mse_by_t else np.zeros((0, H), dtype=np.float64)
    mse_t = mse_by_t.mean(axis=0) if mse_by_t.size else np.zeros((H,), dtype=np.float64)
    rollout_mse_mean = float(mse_t.mean()) if mse_t.size else 0.0
    rollout_mse_last = float(mse_t[-1]) if mse_t.size else 0.0

    metrics = {
        "pred_mse": float(pred_mse),
        "rollout": {"horizon": int(H), "mse_by_t": mse_t.astype(np.float64).tolist(), "mse_mean": float(rollout_mse_mean), "mse_last": float(rollout_mse_last)},
        "diagnostics": {
            "sigma": float(cfg.sigma),
            "top": float(top),
            "top_override": float(cfg.top_override) if cfg.top_override is not None else None,
            "noise_train": None,
            "noise_test": None,
            "achieved": {
                "x_true": _stats_arr(traj),
                "x_noise": _stats_arr(np.zeros((0,), dtype=np.float64)),
                # For this thermal-stream pipeline, reuse the column as a generic scale indicator.
                "x_noise_rel_std": float(np.std(traj, ddof=1) if traj.size >= 2 else 0.0),
                "u_test": _stats_arr(u_test),
                "h_test": _stats_arr(h_test),
                "h_saturation": _sat_fracs(h_test, top=float(top)),
            },
            "dmd": dmd_meta,
            "series_meta": dict(series_meta),
            "r_closure": r_summary,
            "stream": {"n_traj": int(cfg.n_traj), "time_steps": int(cfg.time_steps), "window_L": int(cfg.window_L), "clip_k": float(cfg.clip_k)},
        },
    }
    _write_json(out_dir / "metrics.json", metrics)
    hey = heyting_metrics_from_h(h_test, top=float(top), mutual_thr=float(cfg.mutual_thr), adj_n=int(cfg.adj_n), seed=int(cfg.seed))
    _write_json(out_dir / "heyting_metrics.json", hey)
    _write_json(out_dir / "eigen_spectrum.json", spec)
    _write_json(out_dir / "spectrum_bootstrap.json", boot)
    np.save(out_dir / "koopman_A.npy", A)

    _write_json(
        out_dir / "config.json",
        {
            "version": "thermal_stream_v1",
            "seed": int(cfg.seed),
            "source": str(cfg.source_id),
            "sigma": float(cfg.sigma),
            "stream": {"n_traj": int(cfg.n_traj), "time_steps": int(cfg.time_steps), "window_L": int(cfg.window_L), "clip_k": float(cfg.clip_k)},
        },
    )

    write_manifest_sha256(out_dir / "MANIFEST.sha256", files=list_files_for_manifest(out_dir))
    return metrics
