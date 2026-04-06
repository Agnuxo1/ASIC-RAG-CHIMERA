from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    }


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
class PcChronosPipelineConfig:
    out_dir: Path
    seed: int
    source_id: str
    schema_version: str = "0.1"

    # Trajectory slicing for DMD.
    n_traj: int = 2
    time_steps: int = 96
    train_frac: float = 0.5

    ridge_dmd: float = 1e-6
    rollout_horizon: int = 25

    # Governance.
    top_override: Optional[float] = None

    # Optional iterated closure (R̂) on binarized features.
    r_closure: str = "off"  # off|diag|apply
    r_q: float = 0.8
    r_min_support: int = 20
    r_p_thr: float = 0.9
    r_max_iters: int = 32
    r_graph_path: Optional[Path] = None

    mutual_thr: float = 0.95
    adj_n: int = 4096


def run_pc_chronos_pipeline(
    cfg: PcChronosPipelineConfig,
    *,
    X_features: np.ndarray,
    feature_names: List[str],
    window_t0_ns: np.ndarray,
    window_t1_ns: np.ndarray,
    chronos_summary: Optional[Dict[str, Any]] = None,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = ensure_out_dir(Path(cfg.out_dir))

    X = np.asarray(X_features, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X_features must be (W,K)")
    W, K = int(X.shape[0]), int(X.shape[1])
    if W <= 0 or K <= 0:
        raise ValueError("empty X_features")

    t0 = np.asarray(window_t0_ns, dtype=np.int64).reshape(-1)
    t1 = np.asarray(window_t1_ns, dtype=np.int64).reshape(-1)
    if t0.size != W or t1.size != W:
        raise ValueError("window_t0_ns/window_t1_ns length mismatch")

    n_traj = int(max(2, int(cfg.n_traj)))
    time_steps = int(cfg.time_steps)
    if time_steps <= 1:
        raise ValueError("time_steps must be >= 2")
    n_total = int(n_traj) * int(time_steps)
    if W < n_total:
        raise ValueError(f"insufficient windows: W={W} < n_traj*time_steps={n_total}")

    X_use = X[:n_total, :]
    t0_use = t0[:n_total]
    t1_use = t1[:n_total]

    traj = X_use.reshape(int(n_traj), int(time_steps), int(K))
    n_train = max(1, int(round(float(cfg.train_frac) * int(n_traj))))
    train_traj = traj[:n_train, :, :]
    test_traj = traj[n_train:, :, :]
    if test_traj.size == 0:
        # Ensure at least one test trajectory.
        train_traj = traj[: int(n_traj) - 1, :, :]
        test_traj = traj[int(n_traj) - 1 :, :, :]

    u_train = train_traj.astype(np.float32, copy=False)
    u_test = test_traj.astype(np.float32, copy=False)

    # Optional R̂ closure, with graph optionally frozen from a path.
    r_mode = str(cfg.r_closure).lower().strip()
    r_summary: Optional[Dict[str, Any]] = None
    if r_mode in {"diag", "apply"}:
        from chimera_handoff.pipeline.r_closure import (
            apply_r_closure_diag,
            apply_r_closure_to_u,
            binarize_by_feature_quantile,
            learn_implication_graph,
            load_implication_graph,
            write_implication_graph,
            write_r_closure_timeseries_csv,
        )

        if cfg.r_graph_path is not None:
            adj, thr, mutual_pairs, graph_obj = load_implication_graph(Path(cfg.r_graph_path))
            graph_density = float(graph_obj.get("graph_density", 0.0))
            mutual_edge_count = int(graph_obj.get("mutual_edge_count", 0))
            # Copy the graph JSON into the run dir for provenance.
            _write_json(out_dir / "implication_graph.json", graph_obj)
        else:
            u_cal = u_train.reshape(-1, u_train.shape[-1]).astype(np.float64)
            active, thr = binarize_by_feature_quantile(u_cal, q=float(cfg.r_q))
            graph = learn_implication_graph(active, min_support=int(cfg.r_min_support), p_thr=float(cfg.r_p_thr), laplace_alpha=1.0)
            adj = np.asarray(graph["adj_bool"], dtype=bool)
            mutual_pairs = list(graph.get("mutual_pairs", []))
            graph_density = float(graph.get("graph_density", 0.0))
            mutual_edge_count = int(graph.get("mutual_edge_count", 0))
            write_implication_graph(out_dir / "implication_graph.json", graph, thr=thr, q=float(cfg.r_q))

        u_test_flat = u_test.reshape(-1, u_test.shape[-1]).astype(np.float64)
        stats, rows = apply_r_closure_diag(u_test_flat, thr=thr, adj=adj, mutual_pairs=mutual_pairs, max_iters=int(cfg.r_max_iters))
        r_summary = {
            "mode": str(r_mode),
            "q": float(cfg.r_q),
            "min_support": int(cfg.r_min_support),
            "p_thr": float(cfg.r_p_thr),
            "max_iters": int(cfg.r_max_iters),
            "graph_density": float(graph_density),
            "mutual_edge_count": int(mutual_edge_count),
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

    # Save NPZ contract.
    np.savez_compressed(
        out_dir / "pipeline_inputs.npz",
        X_features=X_use.astype(np.float32, copy=False),
        feature_names=np.asarray([str(s) for s in feature_names], dtype=object),
        window_t0_ns=t0_use.astype(np.int64, copy=False),
        window_t1_ns=t1_use.astype(np.int64, copy=False),
    )
    eigvals, eigvecs = np.linalg.eig(A.astype(np.float64))
    np.savez_compressed(
        out_dir / "pipeline_outputs.npz",
        H=apply_J(traj, top=float(top)).reshape(n_total, K).astype(np.float32, copy=False),
        A_dmd=A.astype(np.float32, copy=False),
        eigvals=np.asarray(eigvals, dtype=np.complex128),
        eigvecs=np.asarray(eigvecs, dtype=np.complex128),
    )

    metrics = {
        "pred_mse": float(pred_mse),
        "rollout": {"horizon": int(H), "mse_by_t": mse_t.astype(np.float64).tolist(), "mse_mean": float(rollout_mse_mean), "mse_last": float(rollout_mse_last)},
        "chronos": dict(chronos_summary) if chronos_summary is not None else None,
        "diagnostics": {
            "schema_version": str(cfg.schema_version),
            "top": float(top),
            "top_override": float(cfg.top_override) if cfg.top_override is not None else None,
            "X_features": _stats_arr(X_use),
            "u_train": _stats_arr(u_train),
            "u_test": _stats_arr(u_test),
            "h_test": _stats_arr(h_test),
            "h_saturation": _sat_fracs(h_test, top=float(top)),
            "dmd": dmd_meta,
            "r_closure": r_summary,
            "protocol": dict(protocol) if protocol is not None else None,
            "stream": {"n_traj": int(n_traj), "time_steps": int(time_steps), "train_frac": float(cfg.train_frac)},
        },
    }
    _write_json(out_dir / "metrics.json", metrics)
    hey = heyting_metrics_from_h(h_test, top=float(top), mutual_thr=float(cfg.mutual_thr), adj_n=int(cfg.adj_n), seed=int(cfg.seed))
    _write_json(out_dir / "heyting_metrics.json", hey)
    _write_json(out_dir / "eigen_spectrum.json", spec)
    np.save(out_dir / "koopman_A.npy", A)

    _write_json(
        out_dir / "config.json",
        {
            "version": "pc_chronos_v0_1",
            "schema_version": str(cfg.schema_version),
            "seed": int(cfg.seed),
            "source": str(cfg.source_id),
            "stream": {"n_traj": int(n_traj), "time_steps": int(time_steps)},
        },
    )

    write_manifest_sha256(out_dir / "MANIFEST.sha256", files=list_files_for_manifest(out_dir))
    return metrics
