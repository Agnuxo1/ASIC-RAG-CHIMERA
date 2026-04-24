from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _bootstrap_ci(diff: np.ndarray, *, n_boot: int = 20000, seed: int = 0) -> Tuple[float, float]:
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


def _median_iqr(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    return float(np.median(x)), float(q3 - q1)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Run root A (e.g., idle)")
    ap.add_argument("--b", required=True, help="Run root B (e.g., cpu_burn)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    a_root = Path(args.a)
    b_root = Path(args.b)
    out = ensure_out_dir(Path(args.out))

    # Prefer summary_prng if present, else require caller to point directly.
    a_csv = a_root / "summary_prng" / "SUMMARY.csv" if (a_root / "summary_prng" / "SUMMARY.csv").exists() else a_root / "SUMMARY.csv"
    b_csv = b_root / "summary_prng" / "SUMMARY.csv" if (b_root / "summary_prng" / "SUMMARY.csv").exists() else b_root / "SUMMARY.csv"
    if not a_csv.exists() or not b_csv.exists():
        raise SystemExit("missing SUMMARY.csv in one of the provided run roots")

    a_rows = _read_csv_rows(a_csv)
    b_rows = _read_csv_rows(b_csv)

    # Index by (source, seed, sigma).
    def key(row: Dict[str, Any]) -> Tuple[str, int, float]:
        return (str(row["source"]), int(row["seed"]), float(row.get("sigma", 0.0)))

    a_map = {key(r): r for r in a_rows}
    b_map = {key(r): r for r in b_rows}
    keys = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not keys:
        raise SystemExit("no overlapping (source,seed,sigma) between runs")

    metrics = [
        "rollout_mse_mean",
        "pred_mse",
        "hey_gap_mean_l1",
        "hey_double_neg_change_frac",
        "hey_mutual_pairs",
        "eig_spectral_radius",
        "eig_neutral_count_0p98_1p02",
        "ach_x_noise_rel_std",
        "ach_h_frac_at_0",
        "ach_h_frac_at_top",
    ]

    by_source: Dict[str, Dict[str, Any]] = {}
    for src in sorted({k[0] for k in keys}):
        src_keys = [k for k in keys if k[0] == src]
        by_source[src] = {"n_pairs": int(len(src_keys)), "by_metric": {}}
        for m in metrics:
            diffs = []
            for k in src_keys:
                a = a_map[k]
                b = b_map[k]
                diffs.append(float(b.get(m, 0.0)) - float(a.get(m, 0.0)))
            d = np.asarray(diffs, dtype=np.float64)
            lo, hi = _bootstrap_ci(d, n_boot=int(args.n_boot), seed=0)
            med, iqr = _median_iqr(d)
            by_source[src]["by_metric"][m] = {"mean_diff": float(d.mean()) if d.size else 0.0, "ci95_lo": float(lo), "ci95_hi": float(hi), "median_diff": float(med), "iqr": float(iqr)}

    report = {"a": str(a_root), "b": str(b_root), "a_summary_csv": str(a_csv), "b_summary_csv": str(b_csv), "by_source": by_source}
    _write_json(out / "REPORT.json", report)

    # Minimal markdown.
    md: List[str] = []
    md.append("# Condition Comparison (paired by seed)")
    md.append("")
    md.append(f"- A: `{a_root}`")
    md.append(f"- B: `{b_root}`")
    md.append(f"- Diff: `B - A` (paired by source/seed)")
    md.append("")
    md.append("| source | n | rollout_mse_mean mean_diff (ci95) | pred_mse mean_diff (ci95) | hey_gap mean_diff (ci95) | dneg_change mean_diff (ci95) | eig_sr mean_diff (ci95) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for src in sorted(by_source.keys()):
        bm = by_source[src]["by_metric"]
        n_pairs = int(by_source[src]["n_pairs"])
        def fmt(x: Dict[str, Any]) -> str:
            return f"{x['mean_diff']:.6g} [{x['ci95_lo']:.6g}, {x['ci95_hi']:.6g}]"
        md.append(
            "| "
            + " | ".join(
                [
                    src,
                    str(n_pairs),
                    fmt(bm["rollout_mse_mean"]),
                    fmt(bm["pred_mse"]),
                    fmt(bm["hey_gap_mean_l1"]),
                    fmt(bm["hey_double_neg_change_frac"]),
                    fmt(bm["eig_spectral_radius"]),
                ]
            )
            + " |"
        )
    md.append("")
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_manifest_sha256(out / "MANIFEST.sha256", files=list_files_for_manifest(out))
    print("wrote:", str(out))


if __name__ == "__main__":
    main()
