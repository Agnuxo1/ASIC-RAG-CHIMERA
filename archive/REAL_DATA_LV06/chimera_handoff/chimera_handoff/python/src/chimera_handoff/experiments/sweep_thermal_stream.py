from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chimera_handoff.entropy.fingerprint import measure_entropy_fingerprint, write_entropy_fingerprint
from chimera_handoff.entropy.health import run_health_tests
from chimera_handoff.entropy.profile import profile_entropy_source, write_entropy_profile
from chimera_handoff.entropy.sources import make_source
from chimera_handoff.entropy.thermal_config import ThermalConfig
from chimera_handoff.entropy.thermal_residual_source import ThermalResidualSource
from chimera_handoff.entropy.thermal_surrogates import SurrogateSpec, apply_surrogate
from chimera_handoff.experiments.thermal_stream_pipeline import ThermalStreamConfig, build_context_features, run_thermal_stream_from_series
from chimera_handoff.pipeline.ratchet import choose_top
from chimera_handoff.system.discovery import discover_entropy_sources
from chimera_handoff.system.report import write_system_profile
from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
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


def _zscore_with_calib(x: np.ndarray, *, calib_n: int, clip_k: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    calib_n = int(min(max(8, int(calib_n)), n))
    c = x[:calib_n].astype(np.float64)
    mu = float(np.mean(c)) if c.size else 0.0
    sd = float(np.std(c, ddof=1) if c.size >= 2 else 0.0)
    sd = float(sd if sd > 1e-9 else 1.0)
    y = ((x.astype(np.float64) - mu) / sd).astype(np.float32)
    y = np.clip(y, -float(clip_k), float(clip_k)).astype(np.float32, copy=False)
    return y, {"normalized": True, "calib_mean": float(mu), "calib_std": float(sd), "clip_k": float(clip_k), "calib_n": int(calib_n)}


def _sha256_quantized(x: np.ndarray, *, q: float = 1e-3) -> str:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return hashlib.sha256(b"").hexdigest()
    qq = float(q)
    if not (qq > 0):
        qq = 1e-3
    z = np.round(x.astype(np.float64) / qq).astype(np.int32)
    return hashlib.sha256(z.tobytes()).hexdigest()


def _generate_iid_series(source_id: str, *, seed: int, n: int) -> np.ndarray:
    src = make_source(source_id, seed=int(seed))
    x = np.asarray(getattr(src, "read_normal_f32")(int(n)), dtype=np.float32).reshape(-1)
    return x


def _collect_thermal_base(
    *,
    seed: int,
    shared_dir: Path,
    cfg: ThermalConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # One telemetry collection per seed; artifacts written under shared_dir/thermal.
    src = ThermalResidualSource(seed=int(seed), config=cfg, out_dir=shared_dir)
    base = src.residual_norm_series()
    meta = src.stream_info()
    t_ns = src.timestamps_ns()
    _write_json(shared_dir / "thermal" / "thermal_series_fingerprint.json", {"sha256_q1e-3": _sha256_quantized(base, q=1e-3), "n": int(base.size)})
    np.savez_compressed(shared_dir / "thermal" / "thermal_series.npz", t_ns=t_ns, resid_norm=base, resid_raw=src.residual_raw_series())
    return base, {"kind": "thermal_residual", "stream_info": meta}


def _make_sources_list(*, include_os_getrandom: bool) -> List[str]:
    base = ["prng", "os_urandom", "timing_jitter", "thermal_residual"]
    if include_os_getrandom:
        base.insert(2, "os_getrandom")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default=None, help="Default: runs/<timestamp>_thermal_stream_v1_2_<tag>")

    # Stream pipeline parameters.
    ap.add_argument("--n-traj", type=int, default=2)
    ap.add_argument("--time-steps", type=int, default=None, help="If omitted, derived from thermal-duration/thermal-dt.")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--window-L", type=int, default=64)
    ap.add_argument("--clip-k", type=float, default=5.0)
    ap.add_argument("--rollout-horizon", type=int, default=25)
    ap.add_argument("--ridge-dmd", type=float, default=1e-6)

    # Thermal telemetry/model parameters.
    ap.add_argument("--thermal-dt", type=float, default=0.05)
    ap.add_argument("--thermal-duration", type=float, default=60.0)
    ap.add_argument("--thermal-calib-seconds", type=float, default=10.0)
    ap.add_argument("--thermal-temp-sensor", default="auto")
    ap.add_argument("--thermal-use-rapl", type=int, default=1)
    ap.add_argument("--thermal-clip-k", type=float, default=5.0)

    # Thermal intervention.
    ap.add_argument("--thermal-intervention", default="none", help="none|cpu_burn|cpu_duty")
    ap.add_argument("--intervention-threads", type=int, default=1)
    ap.add_argument("--intervention-duty", type=float, default=0.5)
    ap.add_argument("--intervention-period", type=float, default=1.0)
    ap.add_argument("--intervention-warmup", type=float, default=0.0)

    # Surrogate config.
    ap.add_argument("--surrogates", default="shuffle,blockshuffle,phase", help="Comma list: shuffle,blockshuffle,phase,iaaft")
    ap.add_argument("--surrogate-block-size", type=int, default=16)
    ap.add_argument("--top-source", default="prng", help="Source used to calibrate top for J per seed (prng|thermal_residual).")

    # R-closure config (optional).
    ap.add_argument("--r-closure", default="off", choices=["off", "diag", "apply"])
    ap.add_argument("--r-q", type=float, default=0.8)
    ap.add_argument("--r-min-support", type=int, default=20)
    ap.add_argument("--r-p-thr", type=float, default=0.9)
    ap.add_argument("--r-max-iters", type=int, default=32)

    # Pre-registration (written into protocol).
    ap.add_argument("--primary-metrics", default="rollout_mse_mean,hey_gap_mean_l1,hey_double_neg_change_frac,eig_spectral_radius")
    ap.add_argument("--primary-effect-threshold", type=float, default=0.0)
    ap.add_argument("--replication-k", type=int, default=3)
    ap.add_argument("--replication-n", type=int, default=4)
    ap.add_argument("--session", "--sessions", dest="session", default=None, help="Optional session tag only (e.g. A or B).")

    args = ap.parse_args()

    seeds = _parse_seeds(str(args.seeds))
    out = Path(args.out) if args.out is not None else Path("runs") / f"{_now_tag()}_thermal_stream_v1_2_{str(args.thermal_intervention).lower()}"
    out = ensure_out_dir(out)

    write_system_profile(out / "system_profile.json", beacon_url=None)
    specs = discover_entropy_sources()
    ids = {s.id: s for s in specs}
    include_getrandom = bool(ids.get("os_getrandom") and ids["os_getrandom"].available)
    sources = _make_sources_list(include_os_getrandom=include_getrandom)
    surrogates = [s.strip().lower() for s in str(args.surrogates).split(",") if s.strip()]
    thermal_surrogate_sources: List[str] = []
    if "shuffle" in surrogates:
        thermal_surrogate_sources.append("thermal_surrogate_shuffle")
    if "blockshuffle" in surrogates:
        thermal_surrogate_sources.append("thermal_surrogate_blockshuffle")
    if "phase" in surrogates:
        thermal_surrogate_sources.append("thermal_surrogate_phase")
    if "iaaft" in surrogates:
        thermal_surrogate_sources.append("thermal_surrogate_iaaft")
    sources = sources + thermal_surrogate_sources

    # If time-steps is not explicitly set, derive it from the telemetry budget so
    # increasing --thermal-duration actually increases data volume.
    expected_samples = int(max(0, round(float(args.thermal_duration) / float(max(1e-6, float(args.thermal_dt))))))
    # Leave slack for dt jitter and scheduling variance; the thermal capture count can be
    # slightly below duration/dt in practice.
    safe_samples = int(max(0, np.floor(0.98 * float(expected_samples))))
    n_traj = int(max(2, int(args.n_traj)))
    window_L = int(args.window_L)
    if safe_samples < 2 * window_L:
        raise SystemExit(f"thermal budget too small: safe_samples={safe_samples} < 2*window_L={2*window_L}")
    if args.time_steps is None:
        # Use as much of the collected series as possible with 2 trajectories, leaving
        # some slack for dt jitter and variability.
        time_steps = int(max(window_L, safe_samples // n_traj))
        # Cap to keep runtime bounded for very long captures.
        time_steps = int(min(time_steps, 2048))
    else:
        time_steps = int(args.time_steps)
        if time_steps < window_L:
            raise SystemExit(f"--time-steps {time_steps} < --window-L {window_L}")

    # Protocol / prereg.
    prereg = {
        "primary_metrics": [s.strip() for s in str(args.primary_metrics).split(",") if s.strip()],
        "primary_effect_threshold": float(args.primary_effect_threshold),
        "replication_rule": {"same_direction_at_least_k": int(args.replication_k), "out_of_n": int(args.replication_n)},
        "notes": ["Effects must separate thermal_residual from its surrogates; prng is a convenience baseline."],
    }
    protocol = {
        "version": "thermal_stream_v1_2",
        "seeds": seeds,
        "sources": sources,
        "session": str(args.session) if args.session else None,
        "stream": {"n_traj": int(n_traj), "time_steps": int(time_steps), "train_frac": float(args.train_frac), "window_L": int(window_L), "clip_k": float(args.clip_k)},
        "thermal": {
            "dt_s": float(args.thermal_dt),
            "duration_s": float(args.thermal_duration),
            "calib_s": float(args.thermal_calib_seconds),
            "temp_sensor": str(args.thermal_temp_sensor),
            "use_rapl": bool(int(args.thermal_use_rapl)),
            "clip_k": float(args.thermal_clip_k),
            "intervention": str(args.thermal_intervention),
            "intervention_threads": int(args.intervention_threads),
            "intervention_duty": float(args.intervention_duty),
            "intervention_period_s": float(args.intervention_period),
            "intervention_warmup_s": float(args.intervention_warmup),
        },
        "surrogate": {"requested": surrogates, "block_size": int(args.surrogate_block_size)},
        "r_closure": {"mode": str(args.r_closure), "q": float(args.r_q), "min_support": int(args.r_min_support), "p_thr": float(args.r_p_thr), "max_iters": int(args.r_max_iters)},
        "preregistration": prereg,
    }
    _write_json(out / "protocol.json", protocol)
    _write_json(out / "preregistered_metrics.json", prereg)
    _write_json(out / "manifest.json", protocol)

    # One-time profiling/fingerprints for byte sources (optional; fast).
    ent_dir = ensure_out_dir(out / "entropy")
    for sid in ["prng", "os_urandom", "timing_jitter"] + (["os_getrandom"] if include_getrandom else []):
        src = make_source(sid, seed=0)
        sample = getattr(src, "read_bytes")(4096)
        health = run_health_tests(sample)
        _write_json(ent_dir / f"entropy_health_{sid}.json", {"source_id": sid, **health.__dict__})
        prof = profile_entropy_source(src, duration_sec=1.0, block_bytes=4096 if sid == "timing_jitter" else 65536)
        write_entropy_profile(ent_dir / f"entropy_profile_{sid}.json", prof)
        fp = measure_entropy_fingerprint(src, duration_sec=1.0)
        write_entropy_fingerprint(ent_dir / f"entropy_fingerprint_{sid}.json", fp)

    runs_root = ensure_out_dir(out / "runs")
    shared_root = ensure_out_dir(out / "thermal_shared")

    n_total = int(n_traj) * int(time_steps)
    calib_n = int(max(8, round(float(args.thermal_calib_seconds) / float(max(1e-6, float(args.thermal_dt))))))

    thermal_cfg = ThermalConfig(
        dt_s=float(args.thermal_dt),
        duration_s=float(args.thermal_duration),
        calib_s=float(args.thermal_calib_seconds),
        temp_sensor=str(args.thermal_temp_sensor),
        use_rapl=bool(int(args.thermal_use_rapl)),
        clip_k=float(args.thermal_clip_k),
        intervention=str(args.thermal_intervention),
        intervention_threads=int(args.intervention_threads),
        intervention_duty=float(args.intervention_duty),
        intervention_period_s=float(args.intervention_period),
        intervention_warmup_s=float(args.intervention_warmup),
    ).sanitized()

    for seed in seeds:
        shared_dir = ensure_out_dir(shared_root / f"seed_{int(seed):02d}")
        thermal_base_raw, thermal_meta = _collect_thermal_base(seed=int(seed), shared_dir=shared_dir, cfg=thermal_cfg)

        # Apply the same final z-score governance to thermal as to other sources.
        thermal_base, thermal_norm_meta = _zscore_with_calib(
            thermal_base_raw,
            calib_n=int(min(int(calib_n), int(thermal_base_raw.size))),
            clip_k=float(args.clip_k),
        )
        thermal_meta = {**thermal_meta, "normalization": dict(thermal_norm_meta)}

        # Precompute thermal variants (for this seed), deterministically, only if requested.
        base_norm, base_norm_meta = thermal_base, thermal_meta
        shuffle = block = phase = iaaft = None
        shuffle_meta: Dict[str, Any] = {}
        block_meta: Dict[str, Any] = {}
        phase_meta: Dict[str, Any] = {}
        iaaft_meta: Dict[str, Any] = {}
        if "shuffle" in surrogates:
            shuffle, shuffle_meta = apply_surrogate(base_norm, SurrogateSpec(kind="shuffle", seed=int(seed) + 1))
        if "blockshuffle" in surrogates:
            block, block_meta = apply_surrogate(base_norm, SurrogateSpec(kind="blockshuffle", seed=int(seed) + 2, block_size=int(args.surrogate_block_size)))
        if "phase" in surrogates:
            phase, phase_meta = apply_surrogate(base_norm, SurrogateSpec(kind="phase", seed=int(seed) + 3))
        if "iaaft" in surrogates:
            iaaft, iaaft_meta = apply_surrogate(base_norm, SurrogateSpec(kind="iaaft", seed=int(seed) + 4))

        # Emit variants fingerprints under shared_dir for audit.
        _write_json(
            shared_dir / "thermal" / "surrogates.json",
            {
                "base": {"sha256_q1e-3": _sha256_quantized(base_norm, q=1e-3), "n": int(base_norm.size)},
                "shuffle": {**shuffle_meta, "sha256_q1e-3": _sha256_quantized(shuffle, q=1e-3), "n": int(shuffle.size)} if shuffle is not None else None,
                "blockshuffle": {**block_meta, "sha256_q1e-3": _sha256_quantized(block, q=1e-3), "n": int(block.size)} if block is not None else None,
                "phase": {**phase_meta, "sha256_q1e-3": _sha256_quantized(phase, q=1e-3), "n": int(phase.size)} if phase is not None else None,
                "iaaft": {**iaaft_meta, "sha256_q1e-3": _sha256_quantized(iaaft, q=1e-3), "n": int(iaaft.size)} if iaaft is not None else None,
            },
        )

        # Calibrate top once per seed on the chosen top-source, then reuse for all sources.
        top_source = str(args.top_source).lower().strip()
        if top_source == "thermal_residual":
            top_series = base_norm[:n_total]
        else:
            raw_prng = _generate_iid_series("prng", seed=int(seed), n=int(n_total))
            top_series, _ = _zscore_with_calib(raw_prng, calib_n=int(min(int(calib_n), int(raw_prng.size))), clip_k=float(args.clip_k))
            top_source = "prng"

        top_traj = top_series[:n_total].reshape(int(n_traj), int(time_steps))
        n_train_top = max(1, int(round(float(args.train_frac) * int(n_traj))))
        top_train = top_traj[:n_train_top, :]
        u_top = build_context_features(top_train, window_L=int(window_L), clip_k=float(args.clip_k))
        cached_top = float(choose_top(u_top, quantile=0.999, min_top=1.0))
        _write_json(
            ensure_out_dir(out / "top_calibration") / f"seed_{int(seed):02d}.json",
            {"top_source": top_source, "top": float(cached_top), "seed": int(seed), "n_traj": int(n_traj), "time_steps": int(time_steps), "n_train_traj": int(n_train_top)},
        )

        # For non-thermal sources: generate iid normals and enforce the same z-score governance.
        for sid in sources:
            run_dir = ensure_out_dir(runs_root / sid / f"seed_{int(seed):02d}")
            if (run_dir / "metrics.json").exists() and (run_dir / "MANIFEST.sha256").exists():
                continue

            if sid == "thermal_residual":
                x = base_norm
                series_meta = {"source_id": sid, "normalized": True, **base_norm_meta, "surrogate": {"kind": "none"}}
            elif sid == "thermal_surrogate_shuffle":
                if shuffle is None:
                    continue
                x = shuffle
                series_meta = {"source_id": sid, "normalized": True, **base_norm_meta, "surrogate": dict(shuffle_meta)}
            elif sid == "thermal_surrogate_blockshuffle":
                if block is None:
                    continue
                x = block
                series_meta = {"source_id": sid, "normalized": True, **base_norm_meta, "surrogate": dict(block_meta)}
            elif sid == "thermal_surrogate_phase":
                if phase is None:
                    continue
                x = phase
                series_meta = {"source_id": sid, "normalized": True, **base_norm_meta, "surrogate": dict(phase_meta)}
            elif sid == "thermal_surrogate_iaaft":
                if iaaft is None:
                    continue
                x = iaaft
                series_meta = {"source_id": sid, "normalized": True, **base_norm_meta, "surrogate": dict(iaaft_meta)}
            else:
                raw = _generate_iid_series(sid, seed=int(seed), n=int(n_total))
                x, norm_meta = _zscore_with_calib(raw, calib_n=int(min(int(calib_n), int(raw.size))), clip_k=float(args.clip_k))
                series_meta = {"source_id": sid, "normalized": True, "surrogate": {"kind": "none"}, "normalization": dict(norm_meta)}
                _write_json(run_dir / "float_series_fingerprint.json", {"sha256_q1e-3": _sha256_quantized(x, q=1e-3), "n": int(x.size)})

            cfg = ThermalStreamConfig(
                out_dir=run_dir,
                seed=int(seed),
                source_id=str(sid),
                sigma=0.0,
                n_traj=int(n_traj),
                time_steps=int(time_steps),
                train_frac=float(args.train_frac),
                window_L=int(window_L),
                clip_k=float(args.clip_k),
                rollout_horizon=int(args.rollout_horizon),
                ridge_dmd=float(args.ridge_dmd),
                top_override=float(cached_top),
                r_closure=str(args.r_closure),
                r_q=float(args.r_q),
                r_min_support=int(args.r_min_support),
                r_p_thr=float(args.r_p_thr),
                r_max_iters=int(args.r_max_iters),
            )
            run_thermal_stream_from_series(cfg, series=x[:n_total], series_meta=series_meta)

    write_manifest_sha256(out / "MANIFEST.sha256", files=list_files_for_manifest(out))
    print("wrote:", str(out))


if __name__ == "__main__":
    main()
