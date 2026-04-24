from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.chronos_metrics import (
    ChronosMetricsConfig,
    compute_chronos_metrics_from_events,
    derive_deltas_csv,
    fit_entropy_bin_edges_from_deltas,
    summarize_chronos_metrics,
    validate_events_timestamps,
)
from chimera_handoff.entropy.chronos_surrogates import SurrogateSpec, apply_surrogate
from chimera_handoff.entropy.pc_pow_share_source import PcPoWShareConfig, PcPoWShareEventSource
from chimera_handoff.entropy.float_contexts import float_context_features, float_context_summary
from chimera_handoff.system.report import write_system_profile
from chimera_handoff.util.manifest import list_files_for_manifest, sha256_file, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(part))
    return out


def _zscore_clip(x: np.ndarray, *, mu: float, sd: float, clip_k: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    sd = float(sd if sd > 1e-9 else 1.0)
    y = (x - float(mu)) / sd
    y = np.clip(y, -float(clip_k), float(clip_k))
    return y.astype(np.float32)

def _sha256_quantized(x: np.ndarray, *, q: float = 1e-6) -> str:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return hashlib.sha256(b"").hexdigest()
    qq = float(q if q > 0 else 1e-6)
    z = np.round(x / qq).astype(np.int64)
    return hashlib.sha256(z.tobytes()).hexdigest()


def _windows_1d(x: np.ndarray, *, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    L = int(L)
    if x.size < L:
        raise ValueError("series shorter than window")
    n = int(x.size) - L + 1
    out = np.empty((n, L), dtype=np.float32)
    for i in range(n):
        out[i] = x[i : i + L]
    return out


def _build_delta_context_features(
    deltas_s: np.ndarray,
    event_t_ns: np.ndarray,
    *,
    window_L: int,
    mu: float,
    sd: float,
    clip_k: float,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Mode A: delta stream -> float contexts (48-d).
    Returns X_features [W,K], feature_names, window_t0_ns, window_t1_ns.
    """
    d = np.asarray(deltas_s, dtype=np.float64).reshape(-1)
    t = np.asarray(event_t_ns, dtype=np.int64).reshape(-1)
    if t.size != d.size + 1:
        raise ValueError("event timestamps must have one more element than deltas")
    z = _zscore_clip(d, mu=float(mu), sd=float(sd), clip_k=float(clip_k))
    W = _windows_1d(z, L=int(window_L))  # (n_win, L)
    k = int(float_context_summary().dim)
    X = np.empty((int(W.shape[0]), k), dtype=np.float32)
    for i in range(int(W.shape[0])):
        X[i] = float_context_features(W[i], clip_k=float(clip_k))
    # Window time span aligns to events [i, i+L].
    w_t0 = t[: int(W.shape[0])].astype(np.int64)
    w_t1 = t[int(window_L) : int(window_L) + int(W.shape[0])].astype(np.int64)
    names = [f"ctx_{j}" for j in range(k)]
    return X, names, w_t0, w_t1


def _read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _build_metric_features(metrics_csv: Path) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Mode B: Chronos metrics rows -> feature vectors.
    Nonnegative features only (for J).
    """
    rows = _read_metrics_csv(Path(metrics_csv))
    if not rows:
        raise ValueError("empty chronos_metrics.csv")

    cols = [
        "cv",
        "hist_entropy",
        "hamming_norm",
        "event_rate_hz",
        "psd_peak_snr_db",
        "psd_peak_hz",
        "psd_peak_hz_error_hz",
    ]
    X = np.asarray([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rows], dtype=np.float64)
    # Enforce nonnegativity for J (and drop NaNs).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # SNR in dB can be negative; keep as a positive indicator.
    X[:, cols.index("psd_peak_snr_db")] = np.maximum(X[:, cols.index("psd_peak_snr_db")], 0.0)
    X = np.maximum(X, 0.0)

    t0 = np.asarray([int(r["window_t0_ns"]) for r in rows], dtype=np.int64)
    t1 = np.asarray([int(r["window_t1_ns"]) for r in rows], dtype=np.int64)
    return X.astype(np.float32), cols, t0, t1


def _save_governance(out_root: Path, *, seed: int, mode: str, top: float, graph_path: Path, delta_norm: Dict[str, Any], entropy_edges: np.ndarray) -> None:
    gov = ensure_out_dir(Path(out_root) / "governance" / f"seed_{int(seed):02d}")
    _write_json(gov / f"top_{mode}.json", {"schema_version": "0.1", "seed": int(seed), "mode": str(mode), "top": float(top)})
    ensure_out_dir(gov / f"r_graph_{mode}")
    shutil.copy2(graph_path, gov / f"r_graph_{mode}" / "implication_graph.json")
    _write_json(gov / "delta_norm.json", {"schema_version": "0.1", **delta_norm})
    _write_json(gov / "entropy_bin_edges.json", {"schema_version": "0.1", "edges": np.asarray(entropy_edges, dtype=np.float64).tolist()})


def _load_governance(root: Path, *, seed: int, mode: str) -> Dict[str, Any]:
    gov = Path(root) / "governance" / f"seed_{int(seed):02d}"
    top = json.loads((gov / f"top_{mode}.json").read_text(encoding="utf-8"))
    delta_norm = json.loads((gov / "delta_norm.json").read_text(encoding="utf-8"))
    edges = json.loads((gov / "entropy_bin_edges.json").read_text(encoding="utf-8"))
    graph = gov / f"r_graph_{mode}" / "implication_graph.json"
    return {"top": float(top["top"]), "delta_norm": delta_norm, "entropy_edges": np.asarray(edges["edges"], dtype=np.float64), "graph_path": graph}


def _write_events_csv(path: Path, *, base_events_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_events_csv, path)


def _build_prereg(
    *,
    protocol_version: str,
    decision_rule: str,
    condition: Optional[str],
    seed_range: str,
    target_hz: float,
) -> Dict[str, Any]:
    proto_v = str(protocol_version).strip()
    if proto_v == "pc_chronos_v0_2":
        return {
            "schema_version": "0.1",
            "protocol_version": "pc_chronos_v0_2",
            "decision_rule": "v0_2_psd_primary",
            "condition": str(condition) if condition is not None else None,
            "seed_range": str(seed_range),
            "target_hz": float(target_hz),
            "primary_metrics": [
                "psd_peak_hz_error_hz_mean",
                "psd_peak_snr_db_mean",
                "psd_peak_q_mean",
                "psd_peak_hz_iqr_hz",
            ],
            "secondary_metrics": [
                "cv_mean",
                "hist_entropy_mean",
            ],
            "replication_rule": {"same_direction_at_least_k": 3, "out_of_n": 4},
            "metric_polarity": {
                "psd_peak_hz_error_hz_mean": -1.0,
                "psd_peak_snr_db_mean": 1.0,
                "psd_peak_q_mean": 1.0,
                "psd_peak_hz_iqr_hz": -1.0,
            },
            "decision_thresholds": {
                "did_vs_surrogates": {"k_of_n": [3, 4], "ci_alpha": 0.05, "bootstrap_resamples": 10000},
                "confound_rejection": {"target_tol_hz": 0.15},
            },
        }

    return {
        "schema_version": "0.1",
        "protocol_version": "pc_chronos_v0_1",
        "decision_rule": str(decision_rule) if decision_rule else "v0_1",
        "condition": str(condition) if condition is not None else None,
        "seed_range": str(seed_range),
        "target_hz": float(target_hz),
        "primary_metrics": [
            "psd_peak_hz_error_hz_mean",
            "psd_peak_snr_db_mean",
            "cv_mean",
            "hist_entropy_mean",
        ],
        "secondary_metrics": [],
        "replication_rule": {"same_direction_at_least_k": 3, "out_of_n": 4},
        "metric_polarity": {
            "psd_peak_hz_error_hz_mean": -1.0,
            "psd_peak_snr_db_mean": 1.0,
            "cv_mean": 1.0,
            "hist_entropy_mean": 1.0,
        },
        "decision_thresholds": {
            "heartbeat_success": {"min_psd_peak_snr_db_increase": 3.0, "max_psd_peak_hz_abs_error_hz": 0.5},
            "did_vs_surrogates": {"k_of_n": [3, 4], "ci_alpha": 0.05, "bootstrap_resamples": 10000},
        },
    }


def _write_prereg(out_root: Path, *, prereg: Dict[str, Any]) -> None:
    _write_json(Path(out_root) / "preregistered_metrics.json", prereg)


def _validate_run_dir(
    run_dir: Path,
    *,
    min_events: int,
    psd_bin_dt_s: float,
    window_events: int,
    require_pipeline: bool,
) -> None:
    events_csv = Path(run_dir) / "events.csv"
    deltas_csv = Path(run_dir) / "deltas.csv"
    meta_p = Path(run_dir) / "chronos_metrics_meta.json"
    if not events_csv.exists():
        raise SystemExit(f"missing {events_csv}")
    if not deltas_csv.exists():
        raise SystemExit(f"missing {deltas_csv}")
    if not meta_p.exists():
        raise SystemExit(f"missing {meta_p}")
    # Monotonic timestamps.
    t = np.asarray([int(r["t_ns"]) for r in _read_metrics_csv(events_csv)], dtype=np.int64)
    validate_events_timestamps(t)
    if int(t.size) < int(min_events):
        raise SystemExit(f"insufficient events in {events_csv}: n_events={int(t.size)} < {int(min_events)}")
    # Delta positivity.
    d = np.asarray([float(r["delta_s"]) for r in _read_metrics_csv(deltas_csv)], dtype=np.float64)
    if d.size and np.any(d <= 0):
        raise SystemExit(f"nonpositive deltas in {deltas_csv}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if str(meta.get("schema_version")) != "0.1":
        raise SystemExit(f"bad schema_version in {meta_p}")
    if int(meta.get("window_events", 0)) != int(window_events):
        raise SystemExit(f"window_events mismatch in {meta_p}")
    if abs(float(meta.get("psd_bin_dt_s", 0.0)) - float(psd_bin_dt_s)) > 1e-9:
        raise SystemExit(f"psd_bin_dt_s mismatch in {meta_p}")
    if require_pipeline:
        if not (Path(run_dir) / "pipeline_inputs.npz").exists():
            raise SystemExit(f"missing {Path(run_dir) / 'pipeline_inputs.npz'}")
        if not (Path(run_dir) / "pipeline_outputs.npz").exists():
            raise SystemExit(f"missing {Path(run_dir) / 'pipeline_outputs.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol-version", default=None, help="If set, tags prereg/protocol (e.g., pc_chronos_v0_2).")
    ap.add_argument("--condition", default=None, choices=[None, "steady", "heartbeat", "batching", "jitter"], help="Alias for intervention/confound preset.")
    ap.add_argument("--allow-seed-reuse", action="store_true", help="Allow v0.2 to reuse v0.1 seeds (default: forbidden).")

    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--difficulty-bits", type=int, default=20)
    ap.add_argument("--backend", default="cpu", choices=["cpu"])
    ap.add_argument("--threads", type=int, default=1)

    ap.add_argument("--intervention", default="none", choices=["none", "heartbeat"])
    ap.add_argument("--heartbeat-hz", type=float, default=2.4)
    ap.add_argument("--heartbeat-duty", type=float, default=0.5)

    ap.add_argument("--confound", default="none", choices=["none", "batching", "jitter"])
    ap.add_argument("--batch-flush-ms", type=float, default=200.0)
    ap.add_argument("--jitter-ms", type=float, default=2.0)

    ap.add_argument("--psd-bin-dt", type=float, default=0.05)
    ap.add_argument("--metrics-window-events", type=int, default=64)
    ap.add_argument("--entropy-bins", type=int, default=32)

    ap.add_argument("--delta-window-L", type=int, default=64)
    ap.add_argument("--clip-k", type=float, default=5.0)
    ap.add_argument("--calib-seconds", type=float, default=10.0)

    ap.add_argument("--r-closure", default="diag", choices=["off", "diag", "apply"])
    ap.add_argument("--r-q", type=float, default=0.8)
    ap.add_argument("--r-min-support", type=int, default=20)
    ap.add_argument("--r-p-thr", type=float, default=0.9)
    ap.add_argument("--r-max-iters", type=int, default=32)

    ap.add_argument("--governance-root", default=None, help="If set, load top + R̂ graph + normalization from this steady root.")
    ap.add_argument("--reuse-raw-root", default=None, help="If set, reuse raw events from another run root (skips hashing).")
    ap.add_argument("--surrogates", default="shuffle,blockshuffle,phase,iaaft")
    ap.add_argument(
        "--pipeline",
        default="off",
        choices=["off", "delta", "metric", "both"],
        help="Default: off (Chronos-only). Set delta/metric/both to run the full J/DMD pipeline (requires extra deps).",
    )
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    # Back-compat aliasing via --condition (spec v0.2 uses this form).
    if args.condition is not None:
        c = str(args.condition).lower().strip()
        if c == "steady":
            args.intervention = "none"
            args.confound = "none"
        elif c == "heartbeat":
            args.intervention = "heartbeat"
            args.confound = "none"
        elif c == "batching":
            args.intervention = "none"
            args.confound = "batching"
        elif c == "jitter":
            args.intervention = "none"
            args.confound = "jitter"

    seeds = _parse_seeds(str(args.seeds))
    proto_v = str(args.protocol_version).strip() if args.protocol_version else "pc_chronos_v0_1"
    if proto_v == "pc_chronos_v0_2" and not bool(args.allow_seed_reuse):
        # v0.1 used seeds 0..9; v0.2 must use fresh seeds.
        if any(0 <= int(s) <= 9 for s in seeds):
            raise SystemExit("pc_chronos_v0_2 forbids seed reuse of 0..9 (use --allow-seed-reuse to override)")
    out_root = Path(args.out) if args.out else Path("runs") / f"{_now_tag()}_pc_chronos_{proto_v}_{str(args.intervention).lower()}_{str(args.confound).lower()}"
    out_root = ensure_out_dir(out_root)

    write_system_profile(out_root / "system_profile.json", beacon_url=None)
    prereg = _build_prereg(
        protocol_version=str(proto_v),
        decision_rule="v0_1",
        condition=str(args.condition) if args.condition is not None else None,
        seed_range=str(args.seeds),
        target_hz=float(args.heartbeat_hz),
    )
    _write_prereg(out_root, prereg=prereg)

    surrogates = [s.strip().lower() for s in str(args.surrogates).split(",") if s.strip()]
    surrogate_kinds = []
    for k in ["shuffle", "blockshuffle", "phase", "iaaft"]:
        if k in surrogates:
            surrogate_kinds.append(k)

    protocol = {
        "schema_version": "0.1",
        "run_kind": str(proto_v),
        "protocol_version": str(proto_v),
        "session": str(args.session) if args.session else None,
        "seeds": seeds,
        "pow": {
            "duration_s": float(args.duration),
            "difficulty_bits": int(args.difficulty_bits),
            "backend": str(args.backend),
            "threads": int(args.threads),
            "intervention": {"mode": str(args.intervention), "heartbeat_hz": float(args.heartbeat_hz), "heartbeat_duty": float(args.heartbeat_duty)},
            "confound": {"mode": str(args.confound), "batch_flush_ms": float(args.batch_flush_ms), "jitter_ms": float(args.jitter_ms)},
        },
        "chronos": {"metrics_window_events": int(args.metrics_window_events), "psd_bin_dt_s": float(args.psd_bin_dt), "entropy_bins": int(args.entropy_bins)},
        "delta_contexts": {"window_L": int(args.delta_window_L), "clip_k": float(args.clip_k), "calib_seconds": float(args.calib_seconds)},
        "surrogates": {"kinds": surrogate_kinds},
        "r_closure": {"mode": str(args.r_closure), "q": float(args.r_q), "min_support": int(args.r_min_support), "p_thr": float(args.r_p_thr), "max_iters": int(args.r_max_iters)},
        "governance_root": str(args.governance_root) if args.governance_root else None,
    }
    _write_json(out_root / "protocol.json", protocol)

    runs_root = ensure_out_dir(out_root / "runs")
    raw_root = ensure_out_dir(out_root / "raw")
    reuse_root = Path(args.reuse_raw_root) if args.reuse_raw_root else None

    for seed in seeds:
        seed_raw = ensure_out_dir(raw_root / f"seed_{int(seed):02d}")

        base_dir = seed_raw / "base"
        if reuse_root is not None:
            src_base = reuse_root / "raw" / f"seed_{int(seed):02d}" / "base"
            if not (src_base / "events.csv").exists():
                raise SystemExit(f"--reuse-raw-root missing {src_base / 'events.csv'}")
            ensure_out_dir(base_dir)
            # Copy base artifacts (events/attempt_rate/etc) for provenance and to make manifests local.
            for p in src_base.glob("*"):
                if p.is_file():
                    shutil.copy2(p, base_dir / p.name)
            base_events_csv = base_dir / "events.csv"
        else:
            # Generate base events (real stream) once per seed.
            src_cfg = PcPoWShareConfig(
                seed=int(seed),
                duration_s=float(args.duration),
                difficulty_bits=int(args.difficulty_bits),
                backend=str(args.backend),
                threads=int(args.threads),
                intervention=str(args.intervention),
                heartbeat_hz=float(args.heartbeat_hz),
                heartbeat_duty=float(args.heartbeat_duty),
                confound=str(args.confound),
                batch_flush_ms=float(args.batch_flush_ms),
                jitter_ms=float(args.jitter_ms),
            ).sanitized()
            base_src = PcPoWShareEventSource(config=src_cfg)
            base_events, _proto = base_src.run(out_dir=base_dir)
            base_events_csv = base_dir / "events.csv"
        # Validate timestamps.
        if reuse_root is not None:
            base_t = np.asarray([int(r["t_ns"]) for r in _read_metrics_csv(base_events_csv)], dtype=np.int64)
        else:
            base_t = np.asarray([int(e.t_ns) for e in base_events], dtype=np.int64)
        validate_events_timestamps(base_t)

        # Prepare surrogates: transform deltas.
        base_deltas_path = seed_raw / "base" / "deltas.csv"
        derive_deltas_csv(base_events_csv, out_path=base_deltas_path)
        drows = _read_metrics_csv(base_deltas_path)
        base_deltas = np.asarray([float(r["delta_s"]) for r in drows], dtype=np.float64)

        # Governance: entropy bin edges and delta normalization.
        # Always measure proximity to the target heartbeat frequency; this stays meaningful across
        # steady/confound conditions (it answers “does the peak lock to the hypothesized Hz?”).
        expected_hb = float(args.heartbeat_hz)
        metrics_cfg = ChronosMetricsConfig(
            window_events=int(args.metrics_window_events),
            psd_bin_dt_s=float(args.psd_bin_dt),
            entropy_bins=int(args.entropy_bins),
        )
        # Entropy bins: load or fit from base deltas (steady run should become the governance root).
        if args.governance_root:
            gov_delta = _load_governance(Path(args.governance_root), seed=int(seed), mode="delta")
            entropy_edges = gov_delta["entropy_edges"]
            delta_norm = gov_delta["delta_norm"]
        else:
            entropy_edges = fit_entropy_bin_edges_from_deltas(base_deltas, n_bins=int(args.entropy_bins), log_delta_clip=float(metrics_cfg.log_delta_clip))
            # Delta normalization from first calib_seconds (approx by time from first event).
            t0 = int(base_t[0]) if base_t.size else 0
            t_cut = t0 + int(round(float(args.calib_seconds) * 1e9))
            idx = np.where(base_t[1:] <= t_cut)[0]
            calib_d = base_deltas[idx] if idx.size else base_deltas[: min(64, int(base_deltas.size))]
            mu = float(np.mean(calib_d)) if calib_d.size else float(np.mean(base_deltas) if base_deltas.size else 0.0)
            sd = float(np.std(calib_d, ddof=1) if calib_d.size >= 2 else 1.0)
            delta_norm = {"seed": int(seed), "mu": float(mu), "sd": float(sd), "clip_k": float(args.clip_k), "calib_seconds": float(args.calib_seconds)}

        # Helper to run one event stream variant through both pipeline modes.
        def run_variant(*, variant_id: str, events_dir: Path, events_csv: Path) -> None:
            # Derive deltas and chronos metrics under this events stream.
            deltas_csv = events_dir / "deltas.csv"
            derive_deltas_csv(events_csv, out_path=deltas_csv)
            metrics_csv = events_dir / "chronos_metrics.csv"
            metrics_meta = events_dir / "chronos_metrics_meta.json"
            compute_chronos_metrics_from_events(
                events_csv,
                out_csv=metrics_csv,
                out_meta=metrics_meta,
                cfg=metrics_cfg,
                entropy_bin_edges=entropy_edges,
                expected_heartbeat_hz=expected_hb,
            )
            chronos_summary = summarize_chronos_metrics(metrics_csv)
            _write_json(events_dir / "chronos_summary.json", {"schema_version": "0.1", **chronos_summary})

            # Always emit Chronos-only run dirs (delta/metric) so downstream tools can operate
            # even when the full J/DMD pipeline is disabled.
            out_delta = ensure_out_dir(runs_root / f"{variant_id}_delta" / f"seed_{int(seed):02d}")
            out_metric = ensure_out_dir(runs_root / f"{variant_id}_metric" / f"seed_{int(seed):02d}")
            for od in [out_delta, out_metric]:
                _write_events_csv(od / "events.csv", base_events_csv=events_csv)
                shutil.copy2(deltas_csv, od / "deltas.csv")
                shutil.copy2(metrics_csv, od / "chronos_metrics.csv")
                shutil.copy2(metrics_meta, od / "chronos_metrics_meta.json")
                shutil.copy2(events_dir / "chronos_summary.json", od / "chronos_summary.json")

            pipeline_mode = str(args.pipeline).lower().strip()
            if pipeline_mode == "off":
                for mode, od in [("delta", out_delta), ("metric", out_metric)]:
                    source_id = f"{variant_id}_{mode}"
                    _write_json(
                        od / "config.json",
                        {"schema_version": "0.1", "seed": int(seed), "source_id": str(source_id), "sigma": 0.0, "chronos_only": True},
                    )
                    _write_json(
                        od / "metrics.json",
                        {
                            "schema_version": "0.1",
                            "source_id": str(source_id),
                            "chronos_only": True,
                            "pred_mse": 0.0,
                            "rollout": {"mse_mean": 0.0, "mse_last": 0.0},
                            "diagnostics": {"sigma": 0.0, "top": 0.0, "achieved": {"h_saturation": {"frac_at_0": 0.0, "frac_at_top": 0.0}}},
                            "chronos": dict(chronos_summary),
                            "protocol_ref": {"run_kind": str(protocol.get("run_kind", "")), "seed": int(seed)},
                        },
                    )
                _validate_run_dir(
                    out_delta,
                    min_events=int(args.metrics_window_events),
                    psd_bin_dt_s=float(args.psd_bin_dt),
                    window_events=int(args.metrics_window_events),
                    require_pipeline=False,
                )
                _validate_run_dir(
                    out_metric,
                    min_events=int(args.metrics_window_events),
                    psd_bin_dt_s=float(args.psd_bin_dt),
                    window_events=int(args.metrics_window_events),
                    require_pipeline=False,
                )
                return

            # Build inputs for Mode A (delta contexts) and Mode B (metric vectors).
            drows2 = _read_metrics_csv(deltas_csv)
            deltas = np.asarray([float(r["delta_s"]) for r in drows2], dtype=np.float64)
            t_ns = np.asarray([int(r["t_ns"]) for r in _read_metrics_csv(events_csv)], dtype=np.int64)
            Xd, names_d, wt0_d, wt1_d = _build_delta_context_features(
                deltas,
                t_ns,
                window_L=int(args.delta_window_L),
                mu=float(delta_norm["mu"]),
                sd=float(delta_norm["sd"]),
                clip_k=float(args.clip_k),
            )
            Xm, names_m, wt0_m, wt1_m = _build_metric_features(metrics_csv)

            # Determine trajectory slicing (W must cover n_traj*time_steps).
            n_traj = 2
            # For both modes, take time_steps as the largest that fits.
            td = int(min(256, max(2, int(Xd.shape[0] // n_traj))))
            tm = int(min(256, max(2, int(Xm.shape[0] // n_traj))))

            # Load governance if provided, or if this run root already has governance (steady baseline).
            gov_root = Path(args.governance_root) if args.governance_root else out_root
            has_gov = (gov_root / "governance" / f"seed_{int(seed):02d}" / "delta_norm.json").exists()
            if args.governance_root or (has_gov and variant_id != "pc_pow_share_events"):
                gov_d = _load_governance(gov_root, seed=int(seed), mode="delta")
                gov_m = _load_governance(gov_root, seed=int(seed), mode="metric")
                top_d = float(gov_d["top"])
                top_m = float(gov_m["top"])
                graph_d = Path(gov_d["graph_path"])
                graph_m = Path(gov_m["graph_path"])
            else:
                top_d = None
                top_m = None
                graph_d = None
                graph_m = None

            # Import pipeline lazily so --pipeline off can run on machines without torch.
            from chimera_handoff.experiments.pc_chronos_pipeline import PcChronosPipelineConfig, run_pc_chronos_pipeline

            if pipeline_mode in {"delta", "both"}:
                cfg_d = PcChronosPipelineConfig(
                    out_dir=out_delta,
                    seed=int(seed),
                    source_id=f"{variant_id}_delta",
                    n_traj=int(n_traj),
                    time_steps=int(td),
                    train_frac=0.5,
                    top_override=float(top_d) if top_d is not None else None,
                    r_closure=str(args.r_closure),
                    r_q=float(args.r_q),
                    r_min_support=int(args.r_min_support),
                    r_p_thr=float(args.r_p_thr),
                    r_max_iters=int(args.r_max_iters),
                    r_graph_path=graph_d,
                )
                run_pc_chronos_pipeline(
                    cfg_d,
                    X_features=Xd,
                    feature_names=names_d,
                    window_t0_ns=wt0_d,
                    window_t1_ns=wt1_d,
                    chronos_summary=chronos_summary,
                    protocol=protocol,
                )

            if pipeline_mode in {"metric", "both"}:
                cfg_m = PcChronosPipelineConfig(
                    out_dir=out_metric,
                    seed=int(seed),
                    source_id=f"{variant_id}_metric",
                    n_traj=int(n_traj),
                    time_steps=int(tm),
                    train_frac=0.5,
                    top_override=float(top_m) if top_m is not None else None,
                    r_closure=str(args.r_closure),
                    r_q=float(args.r_q),
                    r_min_support=int(args.r_min_support),
                    r_p_thr=float(args.r_p_thr),
                    r_max_iters=int(args.r_max_iters),
                    r_graph_path=graph_m,
                )
                run_pc_chronos_pipeline(
                    cfg_m,
                    X_features=Xm,
                    feature_names=names_m,
                    window_t0_ns=wt0_m,
                    window_t1_ns=wt1_m,
                    chronos_summary=chronos_summary,
                    protocol=protocol,
                )

            _validate_run_dir(
                out_delta,
                min_events=int(args.metrics_window_events),
                psd_bin_dt_s=float(args.psd_bin_dt),
                window_events=int(args.metrics_window_events),
                require_pipeline=bool(pipeline_mode in {"delta", "both"}),
            )
            _validate_run_dir(
                out_metric,
                min_events=int(args.metrics_window_events),
                psd_bin_dt_s=float(args.psd_bin_dt),
                window_events=int(args.metrics_window_events),
                require_pipeline=bool(pipeline_mode in {"metric", "both"}),
            )

            # If this is the governance run (steady + real), capture top and graph paths for later.
            if args.governance_root is None and variant_id == "pc_pow_share_events":
                top_d_val = json.loads((out_delta / "metrics.json").read_text(encoding="utf-8"))["diagnostics"]["top"]
                top_m_val = json.loads((out_metric / "metrics.json").read_text(encoding="utf-8"))["diagnostics"]["top"]
                _save_governance(
                    out_root,
                    seed=int(seed),
                    mode="delta",
                    top=float(top_d_val),
                    graph_path=out_delta / "implication_graph.json",
                    delta_norm=delta_norm,
                    entropy_edges=entropy_edges,
                )
                # Metric mode uses its own graph/top; reuse the same delta_norm and entropy edges.
                _save_governance(
                    out_root,
                    seed=int(seed),
                    mode="metric",
                    top=float(top_m_val),
                    graph_path=out_metric / "implication_graph.json",
                    delta_norm=delta_norm,
                    entropy_edges=entropy_edges,
                )

        # Run real stream first (creates governance if this is the baseline root).
        run_variant(variant_id="pc_pow_share_events", events_dir=seed_raw / "base", events_csv=base_events_csv)

        # Now run surrogates, using the base stream's deltas.
        base_t_ns = np.asarray([int(r["t_ns"]) for r in _read_metrics_csv(base_events_csv)], dtype=np.int64)
        base_hashes = [str(r.get("hash_hex", "")) for r in _read_metrics_csv(base_events_csv)]
        base_nonces = [int(r.get("nonce", 0) or 0) for r in _read_metrics_csv(base_events_csv)]

        for kind in surrogate_kinds:
            sdir = ensure_out_dir(seed_raw / f"surrogate_{kind}")
            # Surrogates operate on log10-deltas to preserve positivity after inverse transform.
            # For phase surrogates, we additionally:
            # - clip log-values to a safe range (matches Chronos entropy clipping),
            # - renormalize sum(deltas) to match the base run so the time span stays comparable.
            base_log10 = np.log10(np.maximum(base_deltas, 1e-12)).astype(np.float32)
            y_log10, meta = apply_surrogate(base_log10, SurrogateSpec(kind=str(kind), seed=int(seed) + 7))
            y_log10 = np.asarray(y_log10, dtype=np.float64).reshape(-1)
            y_log10 = np.clip(y_log10, -float(metrics_cfg.log_delta_clip), float(metrics_cfg.log_delta_clip))
            y = np.power(10.0, y_log10, dtype=np.float64)
            sum_base = float(np.sum(base_deltas))
            sum_y = float(np.sum(y))
            if sum_base > 0 and sum_y > 0:
                y = y * (sum_base / sum_y)
            y = np.maximum(y, 1e-12)
            _write_json(
                sdir / "surrogate_fingerprint.json",
                {
                    "schema_version": "0.1",
                    "seed": int(seed),
                    "kind": str(kind),
                    "domain": "log10_delta",
                    "sha256_q1e-6_first64_deltas": _sha256_quantized(y[: min(64, int(y.size))], q=1e-6),
                },
            )
            # Reconstruct surrogate timestamps from surrogate deltas (keep base start time).
            t0 = int(base_t_ns[0]) if base_t_ns.size else 0
            t_surr = np.empty((int(base_t_ns.size),), dtype=np.int64)
            t_surr[0] = t0
            for i in range(1, int(t_surr.size)):
                step_ns = int(max(1, round(float(y[i - 1]) * 1e9)))
                t_surr[i] = int(t_surr[i - 1] + step_ns)
            # Write events.csv for surrogate.
            with (sdir / "events.csv").open("w", encoding="utf-8", newline="") as f:
                import csv

                w = csv.DictWriter(f, fieldnames=["t_ns", "nonce", "hash_hex", "difficulty_bits", "attempts_since_prev", "backend", "notes_json"])
                w.writeheader()
                for i in range(int(t_surr.size)):
                    w.writerow(
                        {
                            "t_ns": int(t_surr[i]),
                            "nonce": int(base_nonces[i]) if i < len(base_nonces) else 0,
                            "hash_hex": str(base_hashes[i]) if i < len(base_hashes) else "",
                            "difficulty_bits": int(args.difficulty_bits),
                            "attempts_since_prev": 0,
                            "backend": "cpu",
                            "notes_json": json.dumps({"surrogate": dict(meta), "domain": "log_delta"}, sort_keys=True),
                        }
                    )
            run_variant(variant_id=f"pc_pow_surrogate_{kind}", events_dir=sdir, events_csv=sdir / "events.csv")

    files = list_files_for_manifest(out_root)
    entries = write_manifest_sha256(out_root / "MANIFEST.sha256", files=files)
    base = out_root.resolve()
    files_json = []
    for e in entries:
        p = (base / e.path).resolve()
        files_json.append({"path": str(e.path), "bytes": int(p.stat().st_size) if p.exists() else 0, "sha256": str(e.sha256)})
    # Aggregate counts if present.
    n_events = 0
    for p in base.rglob("counts.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            n_events += int(obj.get("n_events", 0))
        except Exception:
            continue
    manifest = {
        "schema_version": "0.1",
        "run_id": str(out_root.name),
        "protocol_sha256": sha256_file(out_root / "protocol.json") if (out_root / "protocol.json").exists() else "",
        "files": files_json,
        "counts": {"n_events_total_across_seeds": int(n_events)},
    }
    _write_json(out_root / "manifest.json", manifest)
    from chimera_handoff.schema import validate_run_root

    validate_run_root(out_root)
    print("wrote:", str(out_root))


if __name__ == "__main__":
    main()
