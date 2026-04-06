from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.chronos_metrics import (
    ChronosMetricsConfig,
    compute_chronos_metrics_from_events,
    derive_deltas_csv,
    fit_entropy_bin_edges_from_deltas,
    summarize_chronos_metrics,
)
from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_modes(s: str) -> List[str]:
    modes = [p.strip().lower() for p in str(s).split(",") if p.strip()]
    out: List[str] = []
    for m in modes:
        if m in {"delta", "metric"}:
            out.append(m)
    return out or ["delta", "metric"]


def _iter_seed_dirs(raw_root: Path) -> Iterable[Tuple[int, Path]]:
    for p in sorted(raw_root.glob("seed_*")):
        if not p.is_dir():
            continue
        try:
            seed = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        yield seed, p


def _iter_variants(seed_raw: Path) -> Iterable[Tuple[str, Path]]:
    base = seed_raw / "base"
    if (base / "events.csv").exists():
        yield "pc_pow_share_events", base
    for p in sorted(seed_raw.glob("surrogate_*")):
        if not p.is_dir():
            continue
        kind = p.name.split("_", 1)[1]
        if not (p / "events.csv").exists():
            continue
        yield f"pc_pow_surrogate_{kind}", p


def _ensure_chronos_artifacts(
    events_dir: Path,
    *,
    cfg: ChronosMetricsConfig,
    entropy_bin_edges: Optional[np.ndarray],
    expected_heartbeat_hz: Optional[float],
) -> Dict[str, Any]:
    events_csv = events_dir / "events.csv"
    deltas_csv = events_dir / "deltas.csv"
    metrics_csv = events_dir / "chronos_metrics.csv"
    metrics_meta = events_dir / "chronos_metrics_meta.json"

    if not deltas_csv.exists():
        derive_deltas_csv(events_csv, out_path=deltas_csv)

    if not metrics_csv.exists() or not metrics_meta.exists():
        # If entropy bins are missing, fit from this stream's deltas (best-effort).
        edges = entropy_bin_edges
        if edges is None:
            rows = np.genfromtxt(deltas_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
            d = np.asarray(rows["delta_s"], dtype=np.float64).reshape(-1) if rows.size else np.asarray([], dtype=np.float64)
            edges = fit_entropy_bin_edges_from_deltas(d, n_bins=int(cfg.entropy_bins), log_delta_clip=float(cfg.log_delta_clip))
        compute_chronos_metrics_from_events(
            events_csv,
            out_csv=metrics_csv,
            out_meta=metrics_meta,
            cfg=cfg,
            entropy_bin_edges=edges,
            expected_heartbeat_hz=expected_heartbeat_hz,
        )

    summary = summarize_chronos_metrics(metrics_csv)
    _write_json(events_dir / "chronos_summary.json", {"schema_version": "0.1", **summary})
    return summary


def _write_chronos_only_run(
    run_dir: Path,
    *,
    seed: int,
    source_id: str,
    protocol: Dict[str, Any],
    chronos_summary: Dict[str, Any],
    events_dir: Path,
) -> None:
    ensure_out_dir(run_dir)
    # Attach chronos artifacts for downstream tools (mirrors pipeline-run layout).
    for name in ["events.csv", "deltas.csv", "chronos_metrics.csv", "chronos_metrics_meta.json", "chronos_summary.json"]:
        src = Path(events_dir) / name
        if src.exists():
            (run_dir / name).write_bytes(src.read_bytes())
    _write_json(
        run_dir / "config.json",
        {
            "schema_version": "0.1",
            "seed": int(seed),
            "source_id": str(source_id),
            "sigma": 0.0,
            "chronos_only": True,
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": "0.1",
            "source_id": str(source_id),
            "chronos_only": True,
            "pred_mse": 0.0,
            "rollout": {"mse_mean": 0.0, "mse_last": 0.0},
            "diagnostics": {
                "sigma": 0.0,
                "top": 0.0,
                "achieved": {"h_saturation": {"frac_at_0": 0.0, "frac_at_top": 0.0}},
            },
            "chronos": dict(chronos_summary),
            "protocol_ref": {"run_kind": str(protocol.get("run_kind", "")), "seed": int(seed)},
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="run_root", required=True)
    ap.add_argument("--modes", default="delta,metric", help="Comma list: delta,metric.")
    ap.add_argument("--expected-heartbeat-hz", type=float, default=None, help="If set, computes peak error vs this frequency.")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    raw_root = run_root / "raw"
    if not raw_root.exists():
        raise SystemExit(f"missing {raw_root}")

    protocol_p = run_root / "protocol.json"
    protocol = _read_json(protocol_p) if protocol_p.exists() else {}

    chron = dict(protocol.get("chronos", {}))
    cfg = ChronosMetricsConfig(
        window_events=int(chron.get("metrics_window_events", 64)),
        psd_bin_dt_s=float(chron.get("psd_bin_dt_s", 0.05)),
        entropy_bins=int(chron.get("entropy_bins", 32)),
    )
    expected_hb = float(args.expected_heartbeat_hz) if args.expected_heartbeat_hz is not None else None
    modes = _parse_modes(str(args.modes))

    runs_root = ensure_out_dir(run_root / "runs")

    # Optional: derive entropy bin edges from the base stream (seed 0 preferred).
    entropy_edges: Optional[np.ndarray] = None
    for seed, seed_raw in _iter_seed_dirs(raw_root):
        base = seed_raw / "base" / "deltas.csv"
        if base.exists():
            rows = np.genfromtxt(base, delimiter=",", names=True, dtype=None, encoding="utf-8")
            d = np.asarray(rows["delta_s"], dtype=np.float64).reshape(-1) if rows.size else np.asarray([], dtype=np.float64)
            if d.size:
                entropy_edges = fit_entropy_bin_edges_from_deltas(d, n_bins=int(cfg.entropy_bins), log_delta_clip=float(cfg.log_delta_clip))
            break

    wrote = 0
    for seed, seed_raw in _iter_seed_dirs(raw_root):
        for variant_id, events_dir in _iter_variants(seed_raw):
            chronos_summary = _ensure_chronos_artifacts(
                events_dir,
                cfg=cfg,
                entropy_bin_edges=entropy_edges,
                expected_heartbeat_hz=expected_hb,
            )
            for mode in modes:
                source = f"{variant_id}_{mode}"
                out_dir = runs_root / source / f"seed_{int(seed):02d}"
                _write_chronos_only_run(
                    out_dir,
                    seed=int(seed),
                    source_id=source,
                    protocol=protocol,
                    chronos_summary=chronos_summary,
                    events_dir=events_dir,
                )
                wrote += 1

    files = list_files_for_manifest(run_root)
    entries = write_manifest_sha256(run_root / "MANIFEST.sha256", files=files)
    files_json = [{"path": str(e.path), "sha256": str(e.sha256)} for e in entries]
    _write_json(
        run_root / "manifest.json",
        {"schema_version": "0.1", "run_id": str(run_root.name), "files": files_json, "counts": {"n_runs_written": int(wrote)}},
    )
    print(f"ingested chronos-only runs: wrote={wrote} root={run_root}")


if __name__ == "__main__":
    main()
