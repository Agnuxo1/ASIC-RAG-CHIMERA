from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, *, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _median_iqr(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    return float(np.median(x)), float(q3 - q1)


def _trimmed_mean(x: np.ndarray, *, trim: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    t = float(trim)
    lo = float(np.quantile(x, t))
    hi = float(np.quantile(x, 1.0 - t))
    y = x[(x >= lo) & (x <= hi)]
    return float(y.mean()) if y.size else float(x.mean())


def _bootstrap_ci(diff: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    n = int(diff.size)
    stats = np.empty((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        s = diff[rng.integers(0, n, size=n)]
        stats[i] = float(np.mean(s))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _read_psd_hz_error_mean(seed_dir: Path, *, target_hz: Optional[float]) -> Optional[float]:
    if target_hz is None or not np.isfinite(float(target_hz)):
        return None
    p = Path(seed_dir) / "chronos_metrics.csv"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            hz = np.asarray([float(row.get("psd_peak_hz", 0.0) or 0.0) for row in r], dtype=np.float64)
        if hz.size == 0:
            return None
        return float(np.mean(np.abs(hz - float(target_hz))))
    except Exception:
        return None


def _collect_runs(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs = run_root / "runs"
    if not runs.exists():
        raise SystemExit(f"missing {runs} (expected sweep output)")
    protocol = _read_json(run_root / "protocol.json") if (run_root / "protocol.json").exists() else {}
    target_hz = None
    try:
        target_hz = float(dict(dict(protocol.get("pow", {})).get("intervention", {})).get("heartbeat_hz"))
    except Exception:
        target_hz = None
    for source_dir in sorted([p for p in runs.iterdir() if p.is_dir()]):
        source = source_dir.name
        for seed_dir in sorted([p for p in source_dir.glob("seed_*") if p.is_dir()]):
            seed = int(seed_dir.name.split("_")[1])
            cfg_p = seed_dir / "config.json"
            metrics_p = seed_dir / "metrics.json"
            hey_p = seed_dir / "heyting_metrics.json"
            eig_p = seed_dir / "eigen_spectrum.json"
            boot_p = seed_dir / "spectrum_bootstrap.json"
            rstats_p = seed_dir / "r_closure_stats.json"
            if not metrics_p.exists():
                continue
            cfg = _read_json(cfg_p) if cfg_p.exists() else {}
            m = _read_json(metrics_p)
            hey = _read_json(hey_p) if hey_p.exists() else {}
            eig = _read_json(eig_p) if eig_p.exists() else {}
            boot = _read_json(boot_p) if boot_p.exists() else {}
            rstats = _read_json(rstats_p) if rstats_p.exists() else {}
            ach = dict(m.get("diagnostics", {}).get("achieved", {}))
            sat = dict(ach.get("h_saturation", {}))
            chronos = dict(m.get("chronos") or {})
            # Ensure error-to-target is available even when the run was generated without expected_heartbeat_hz.
            psd_err_override = _read_psd_hz_error_mean(seed_dir, target_hz=target_hz)
            rows.append(
                {
                    "seed": int(seed),
                    "source": str(source),
                    "sigma": float(cfg.get("sigma", m.get("diagnostics", {}).get("sigma", 0.0))),
                    "pred_mse": float(m.get("pred_mse", 0.0)),
                    "rollout_mse_mean": float(m.get("rollout", {}).get("mse_mean", 0.0)),
                    "rollout_mse_last": float(m.get("rollout", {}).get("mse_last", 0.0)),
                    "cv_mean": float(chronos.get("cv_mean", 0.0)),
                    "hist_entropy_mean": float(chronos.get("hist_entropy_mean", 0.0)),
                    "psd_peak_snr_db_mean": float(chronos.get("psd_peak_snr_db_mean", 0.0)),
                    "psd_peak_hz_median": float(chronos.get("psd_peak_hz_median", 0.0)),
                    "psd_peak_hz_error_hz_mean": float(psd_err_override) if psd_err_override is not None else float(chronos.get("psd_peak_hz_error_hz_mean", 0.0)),
                    "psd_peak_q_mean": float(chronos.get("psd_peak_q_mean", 0.0)),
                    "psd_peak_hz_iqr_hz": float(chronos.get("psd_peak_hz_iqr_hz", 0.0)),
                    "hey_gap_mean_l1": float(hey.get("regularity_gap", {}).get("mean_l1", 0.0)),
                    "hey_double_neg_change_frac": float(hey.get("double_neg_change_frac", 0.0)),
                    "hey_mutual_pairs": int(hey.get("implication", {}).get("mutual_pairs_count", 0)),
                    "hey_anomaly_mean": float(hey.get("anomaly", {}).get("score_l1_mean", 0.0)),
                    "eig_abs_mean": float(eig.get("eig_abs_mean", 0.0)),
                    "eig_spectral_radius": float(eig.get("spectral_radius", 0.0)),
                    "eig_neutral_count_0p98_1p02": int(eig.get("neutral_count_abs_0p98_1p02", 0)),
                    "eig_neutral_count_0p95_1p05": int(eig.get("neutral_count_abs_0p95_1p05", 0)),
                    "eig_neutral_angle_abs_mean": float(eig.get("neutral_angle_abs_mean_in_0p98_1p02", 0.0)),
                    "boot_spectral_radius_p50": float(dict(boot.get("spectral_radius", {})).get("p50", 0.0)),
                    "boot_neutral_0p98_1p02_p50": float(dict(boot.get("neutral_count_abs_0p98_1p02", {})).get("p50", 0.0)),
                    "ach_x_noise_rel_std": float(ach.get("x_noise_rel_std", 0.0)),
                    "ach_h_frac_at_0": float(sat.get("frac_at_0", 0.0)),
                    "ach_h_frac_at_top": float(sat.get("frac_at_top", 0.0)),
                    "r_birth_p90": float(rstats.get("birth_p90", 0.0)),
                    "r_mutual_edge_count": float(rstats.get("mutual_edge_count", 0.0)),
                    "r_mutual_active_pairs_rate_mean": float(rstats.get("mutual_active_pairs_rate_mean", 0.0)),
                }
            )
    return rows


def _plot_box(out_path: Path, *, data: Dict[str, List[float]], title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    labels = list(data.keys())
    vals = [data[k] for k in labels]
    fig = plt.figure(figsize=(9, 4))
    try:
        plt.boxplot(vals, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(vals, labels=labels, showmeans=True)
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="run_root", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--baseline-source", default=None, help="Override baseline source for paired effects (default: prng if present else first source).")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--catastrophic-rollout", type=float, default=1000.0)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = ensure_out_dir(Path(args.out_dir))
    protocol = _read_json(run_root / "protocol.json") if (run_root / "protocol.json").exists() else {}
    run_kind = str(protocol.get("run_kind", "")).strip().lower()
    is_pc_chronos = run_kind.startswith("pc_chronos")

    rows = _collect_runs(run_root)
    if not rows:
        raise SystemExit(f"no runs found under {run_root}/runs")

    _write_csv(
        out_dir / "SUMMARY.csv",
        rows=rows,
        fieldnames=[
            "seed",
            "source",
            "sigma",
            "pred_mse",
            "rollout_mse_mean",
            "rollout_mse_last",
            "cv_mean",
            "hist_entropy_mean",
            "psd_peak_snr_db_mean",
            "psd_peak_hz_median",
            "psd_peak_hz_error_hz_mean",
            "psd_peak_q_mean",
            "psd_peak_hz_iqr_hz",
            "hey_gap_mean_l1",
            "hey_double_neg_change_frac",
            "hey_mutual_pairs",
            "hey_anomaly_mean",
            "eig_abs_mean",
            "eig_spectral_radius",
            "eig_neutral_count_0p98_1p02",
            "eig_neutral_count_0p95_1p05",
            "eig_neutral_angle_abs_mean",
            "boot_spectral_radius_p50",
            "boot_neutral_0p98_1p02_p50",
            "ach_x_noise_rel_std",
            "ach_h_frac_at_0",
            "ach_h_frac_at_top",
            "r_birth_p90",
            "r_mutual_edge_count",
            "r_mutual_active_pairs_rate_mean",
        ],
    )

    # Aggregate by source.
    sources = sorted({r["source"] for r in rows})
    if args.baseline_source is not None:
        baseline = str(args.baseline_source)
        if baseline not in sources:
            raise SystemExit(f"--baseline-source {baseline!r} not found in sources={sources}")
    else:
        baseline = "prng" if "prng" in sources else sources[0]

    sigmas = sorted({float(r.get("sigma", 0.0)) for r in rows})
    report: Dict[str, Any] = {"baseline_source": baseline, "sigmas": sigmas, "by_sigma": {}}
    metrics = [
        "rollout_mse_mean",
        "pred_mse",
        "cv_mean",
        "hist_entropy_mean",
        "psd_peak_snr_db_mean",
        "psd_peak_hz_median",
        "psd_peak_hz_error_hz_mean",
        "psd_peak_q_mean",
        "psd_peak_hz_iqr_hz",
        "hey_gap_mean_l1",
        "hey_mutual_pairs",
        "hey_double_neg_change_frac",
        "eig_spectral_radius",
        "eig_neutral_count_0p98_1p02",
        "eig_neutral_angle_abs_mean",
        "boot_spectral_radius_p50",
        "boot_neutral_0p98_1p02_p50",
        "ach_x_noise_rel_std",
        "ach_h_frac_at_0",
        "ach_h_frac_at_top",
        "r_birth_p90",
        "r_mutual_edge_count",
        "r_mutual_active_pairs_rate_mean",
    ]
    for sigma in sigmas:
        key = f"{sigma:g}"
        report["by_sigma"][key] = {"sigma": float(sigma), "sources": {}}
        for src in sources:
            report["by_sigma"][key]["sources"][src] = {}
            for metric in metrics:
                vals = np.asarray([float(r[metric]) for r in rows if r["source"] == src and float(r.get("sigma", 0.0)) == float(sigma)], dtype=np.float64)
                med, iqr = _median_iqr(vals)
                tmean = _trimmed_mean(vals, trim=float(args.trim_frac))
                catastrophic = int(np.sum(~np.isfinite(vals) | (vals > float(args.catastrophic_rollout)))) if metric.startswith("rollout") else 0
                report["by_sigma"][key]["sources"][src][metric] = {
                    "n": int(vals.size),
                    "mean": float(vals.mean()) if vals.size else 0.0,
                    "median": float(med),
                    "iqr": float(iqr),
                    "trimmed_mean": float(tmean),
                    "catastrophic": int(catastrophic),
                }

    # Paired effects vs baseline for rollout_mse_mean.
    effects_by_sigma: Dict[str, List[Dict[str, Any]]] = {}
    for sigma in sigmas:
        key = f"{sigma:g}"
        effects: List[Dict[str, Any]] = []
        base_by_seed = {int(r["seed"]): r for r in rows if r["source"] == baseline and float(r.get("sigma", 0.0)) == float(sigma)}
        for src in sources:
            if src == baseline:
                continue
            other_by_seed = {int(r["seed"]): r for r in rows if r["source"] == src and float(r.get("sigma", 0.0)) == float(sigma)}
            diffs = []
            used = []
            for seed, br in base_by_seed.items():
                orow = other_by_seed.get(seed)
                if orow is None:
                    continue
                diffs.append(float(orow["rollout_mse_mean"]) - float(br["rollout_mse_mean"]))
                used.append(int(seed))
            diff = np.asarray(diffs, dtype=np.float64)
            lo, hi = _bootstrap_ci(diff, n_boot=int(args.n_boot), seed=0)
            effects.append(
                {
                    "source": src,
                    "n": int(diff.size),
                    "mean_diff": float(diff.mean()) if diff.size else 0.0,
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                    "seeds": used,
                }
            )
        effects_by_sigma[key] = effects
    report["paired_effects_vs_baseline_rollout_mse_mean_by_sigma"] = effects_by_sigma

    _write_json(out_dir / "REPORT.json", report)

    # Plots.
    for sigma in sigmas:
        key = f"{sigma:g}"
        if is_pc_chronos:
            for metric, ylabel in [
                ("psd_peak_hz_median", "psd_peak_hz_median"),
                ("psd_peak_hz_error_hz_mean", "psd_peak_hz_error_hz_mean"),
                ("psd_peak_snr_db_mean", "psd_peak_snr_db_mean"),
                ("psd_peak_q_mean", "psd_peak_q_mean"),
                ("psd_peak_hz_iqr_hz", "psd_peak_hz_iqr_hz"),
            ]:
                box_data: Dict[str, List[float]] = {}
                for src in sources:
                    vals = [float(r.get(metric, 0.0)) for r in rows if r["source"] == src and float(r.get("sigma", 0.0)) == float(sigma)]
                    vals = [v for v in vals if np.isfinite(v)]
                    if vals:
                        box_data[src] = vals
                _plot_box(
                    out_dir / "PLOTS" / f"{metric}_by_source_sigma_{key}.png",
                    data=box_data,
                    title=f"PC-CHRONOS {metric} by source (sigma={sigma:g})",
                    ylabel=ylabel,
                )
        else:
            box_data: Dict[str, List[float]] = {}
            for src in sources:
                vals = [float(r["rollout_mse_mean"]) for r in rows if r["source"] == src and float(r.get("sigma", 0.0)) == float(sigma)]
                if vals:
                    box_data[src] = vals
            _plot_box(
                out_dir / "PLOTS" / f"rollout_mse_mean_by_source_sigma_{key}.png",
                data=box_data,
                title=f"rollout_mse_mean by source (sigma={sigma:g})",
                ylabel="rollout_mse_mean",
            )

    md: List[str] = []
    if is_pc_chronos:
        md.append("# PC-CHRONOS — Sweep Summary")
        md.append("")
        md.append(f"- Run root: `{run_root}`")
        md.append(f"- Baseline source: `{baseline}`")
        md.append("")
        for sigma in sigmas:
            key = f"{sigma:g}"
            md.append(f"## Sigma = {sigma:g}")
            md.append("")
            md.append(
                "| source | n | psd_peak_hz_median (median±iqr) | psd_peak_hz_error_hz_mean (median±iqr) | psd_peak_snr_db_mean (median±iqr) | psd_peak_q_mean (median±iqr) | psd_peak_hz_iqr_hz (median±iqr) |"
            )
            md.append("|---|---:|---:|---:|---:|---:|---:|")
            for src in sources:
                r_hz = report["by_sigma"][key]["sources"][src]["psd_peak_hz_median"]
                r_err = report["by_sigma"][key]["sources"][src]["psd_peak_hz_error_hz_mean"]
                r_snr = report["by_sigma"][key]["sources"][src]["psd_peak_snr_db_mean"]
                r_q = report["by_sigma"][key]["sources"][src]["psd_peak_q_mean"]
                r_iqr = report["by_sigma"][key]["sources"][src]["psd_peak_hz_iqr_hz"]
                md.append(
                    f"| {src} | {r_hz['n']} | {r_hz['median']:.6g}±{r_hz['iqr']:.6g} | {r_err['median']:.6g}±{r_err['iqr']:.6g} | {r_snr['median']:.6g}±{r_snr['iqr']:.6g} | {r_q['median']:.6g}±{r_q['iqr']:.6g} | {r_iqr['median']:.6g}±{r_iqr['iqr']:.6g} |"
                )
            md.append("")
    else:
        md.append("# CHIMERA — Sweep Summary")
        md.append("")
        md.append(f"- Run root: `{run_root}`")
        md.append(f"- Baseline source: `{baseline}`")
        md.append("")
        md.append("_Non-PC-CHRONOS run kind: report table omitted in handoff build._")
        md.append("")
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_manifest_sha256(out_dir / "MANIFEST.sha256", files=list_files_for_manifest(out_dir))
    print("wrote:", str(out_dir))


if __name__ == "__main__":
    main()
