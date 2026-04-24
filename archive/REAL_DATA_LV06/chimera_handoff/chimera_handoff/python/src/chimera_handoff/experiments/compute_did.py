from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chimera_handoff.util.manifest import list_files_for_manifest, write_manifest_sha256
from chimera_handoff.util.paths import ensure_out_dir


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bootstrap_ci(diff: np.ndarray, *, n_boot: int, seed: int = 0) -> Tuple[float, float]:
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


def _paired_cohens_d(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    if diff.size < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def _resolve_metric_name(name: str) -> str:
    n = str(name).strip()
    m = n.lower()
    mapping = {
        "hey_gap": "hey_gap_mean_l1",
        "gap": "hey_gap_mean_l1",
        "dneg_change": "hey_double_neg_change_frac",
        "dneg": "hey_double_neg_change_frac",
        "eig_sr": "eig_spectral_radius",
        "sr": "eig_spectral_radius",
        "cv": "cv_mean",
        "hist_entropy": "hist_entropy_mean",
        "psd_peak_snr": "psd_peak_snr_db_mean",
        "psd_snr": "psd_peak_snr_db_mean",
        "psd_peak_hz_error": "psd_peak_hz_error_hz_mean",
        "psd_hz_error": "psd_peak_hz_error_hz_mean",
        "psd_peak_hz": "psd_peak_hz_median",
        "psd_peak_q": "psd_peak_q_mean",
        "psd_q": "psd_peak_q_mean",
        "psd_peak_hz_iqr": "psd_peak_hz_iqr_hz",
        "psd_hz_iqr": "psd_peak_hz_iqr_hz",
        "birth_p90": "r_birth_p90",
        "r_birth_p90": "r_birth_p90",
        "mutual_edge_count": "r_mutual_edge_count",
        "r_mutual_edge_count": "r_mutual_edge_count",
        "mutual_active_pairs_rate": "r_mutual_active_pairs_rate_mean",
        "mutual_active_rate": "r_mutual_active_pairs_rate_mean",
        "r_mutual_active_pairs_rate_mean": "r_mutual_active_pairs_rate_mean",
    }
    return mapping.get(m, n)


def _find_summary_csv(root: Path) -> Path:
    # Prefer the summary produced by summarize (baseline=prng) if present.
    p = root / "summary_prng" / "SUMMARY.csv"
    if p.exists():
        return p
    p = root / "SUMMARY.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"missing SUMMARY.csv under {root}")


def _index_rows(rows: List[Dict[str, Any]], *, metrics: Sequence[str]) -> Dict[Tuple[str, int, float], Dict[str, float]]:
    out: Dict[Tuple[str, int, float], Dict[str, float]] = {}
    for r in rows:
        src = str(r["source"])
        seed = int(r["seed"])
        sigma = float(r.get("sigma", 0.0))
        d: Dict[str, float] = {}
        for m in metrics:
            d[m] = float(r.get(m, 0.0) or 0.0)
        out[(src, seed, sigma)] = d
    return out


def _paired_diff(
    a: Dict[Tuple[str, int, float], Dict[str, float]],
    b: Dict[Tuple[str, int, float], Dict[str, float]],
    *,
    src: str,
    metric: str,
    sigma: float = 0.0,
) -> Tuple[np.ndarray, List[int]]:
    diffs: List[float] = []
    used: List[int] = []
    for (s, seed, sg), va in a.items():
        if s != src or float(sg) != float(sigma):
            continue
        vb = b.get((s, seed, float(sigma)))
        if vb is None:
            continue
        diffs.append(float(vb[metric]) - float(va[metric]))
        used.append(int(seed))
    return np.asarray(diffs, dtype=np.float64), used


def _paired_diff_between_sources(
    rows: Dict[Tuple[str, int, float], Dict[str, float]],
    *,
    src_a: str,
    src_b: str,
    metric: str,
    sigma: float = 0.0,
) -> Tuple[np.ndarray, List[int]]:
    diffs: List[float] = []
    used: List[int] = []
    for (s, seed, sg), va in rows.items():
        if s != src_a or float(sg) != float(sigma):
            continue
        vb = rows.get((src_b, int(seed), float(sigma)))
        if vb is None:
            continue
        diffs.append(float(va[metric]) - float(vb[metric]))
        used.append(int(seed))
    return np.asarray(diffs, dtype=np.float64), used


def _summarize_diff(diff: np.ndarray, *, n_boot: int) -> Dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    lo, hi = _bootstrap_ci(diff, n_boot=int(n_boot), seed=0)
    mean = float(np.mean(diff)) if diff.size else 0.0
    return {
        "n": int(diff.size),
        "mean": float(mean),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "cohens_d_paired": float(_paired_cohens_d(diff)),
        "ci_excludes_0": bool((lo > 0.0) or (hi < 0.0)),
        "direction": "pos" if mean > 0 else ("neg" if mean < 0 else "zero"),
    }

def _decide_k_of_n_same_direction(
    summaries: Dict[str, Dict[str, Any]],
    *,
    metrics: Sequence[str],
    k: int,
    polarity: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    metrics = [str(m) for m in metrics]
    k = int(k)
    passed: List[str] = []
    dirs: List[str] = []
    for m in metrics:
        s = summaries.get(m)
        if not s:
            continue
        if bool(s.get("ci_excludes_0")):
            passed.append(m)
            pol = float((polarity or {}).get(m, 1.0))
            mean = float(s.get("mean", 0.0)) * pol
            dirs.append("pos" if mean > 0 else ("neg" if mean < 0 else "zero"))
    nonzero_dirs = [d for d in dirs if d in {"pos", "neg"}]
    same_dir = (len(set(nonzero_dirs)) == 1) if nonzero_dirs else False
    direction = nonzero_dirs[0] if same_dir else "mixed"
    ok = (len(passed) >= k) and same_dir
    return {"pass": bool(ok), "k": int(k), "n": int(len(metrics)), "passed": passed, "direction": direction, "polarity_applied": bool(polarity)}

def _decide_k_of_n_ci_excludes_0(
    summaries: Dict[str, Dict[str, Any]],
    *,
    metrics: Sequence[str],
    k: int,
) -> Dict[str, Any]:
    metrics = [str(m) for m in metrics]
    k = int(k)
    passed: List[str] = []
    for m in metrics:
        s = summaries.get(m)
        if not s:
            continue
        if bool(s.get("ci_excludes_0")):
            passed.append(m)
    ok = len(passed) >= k
    return {"pass": bool(ok), "k": int(k), "n": int(len(metrics)), "passed": passed}


def _decide_k_of_n_matches_polarity(
    summaries: Dict[str, Dict[str, Any]],
    *,
    metrics: Sequence[str],
    k: int,
    polarity: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    metrics = [str(m) for m in metrics]
    k = int(k)
    passed: List[str] = []
    for m in metrics:
        s = summaries.get(m)
        if not s:
            continue
        if not bool(s.get("ci_excludes_0")):
            continue
        pol = float((polarity or {}).get(m, 1.0))
        mean = float(s.get("mean", 0.0))
        if mean * pol > 0:
            passed.append(m)
    ok = len(passed) >= k
    return {"pass": bool(ok), "k": int(k), "n": int(len(metrics)), "passed": passed, "polarity_applied": bool(polarity)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", required=True)
    ap.add_argument("--burn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--treatment-source", default="thermal_residual", help="Primary treated source (default: thermal_residual).")
    ap.add_argument("--metrics", default="rollout_mse_mean,hey_gap,dneg_change,eig_sr")
    ap.add_argument("--surrogates", default="thermal_surrogate_shuffle,thermal_surrogate_blockshuffle,thermal_surrogate_phase,thermal_surrogate_iaaft")
    ap.add_argument("--decision-rule", default="default", choices=["default", "v0_2_psd_primary"], help="Decision logic profile.")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    idle_root = Path(args.idle)
    burn_root = Path(args.burn)
    out = ensure_out_dir(Path(args.out))

    metrics = [_resolve_metric_name(m) for m in str(args.metrics).split(",") if m.strip()]
    surrogates = [s.strip() for s in str(args.surrogates).split(",") if s.strip()]

    idle_csv = _find_summary_csv(idle_root)
    burn_csv = _find_summary_csv(burn_root)
    idle_rows = _read_csv_rows(idle_csv)
    burn_rows = _read_csv_rows(burn_csv)
    idle = _index_rows(idle_rows, metrics=metrics)
    burn = _index_rows(burn_rows, metrics=metrics)

    # Load preregistration if present (for annotation).
    prereg = None
    for p in [idle_root / "preregistered_metrics.json", burn_root / "preregistered_metrics.json"]:
        if p.exists():
            prereg = json.loads(p.read_text(encoding="utf-8"))
            break

    thermal = str(args.treatment_source).strip()
    primary_metrics: List[str] = []
    replication_k = 3
    metric_polarity: Dict[str, float] = {}
    if isinstance(prereg, dict):
        primary_metrics = [_resolve_metric_name(m) for m in prereg.get("primary_metrics", []) if str(m).strip()]
        rr = prereg.get("replication_rule", {})
        if isinstance(rr, dict):
            replication_k = int(rr.get("same_direction_at_least_k", replication_k))
        mp = prereg.get("metric_polarity", {})
        if isinstance(mp, dict):
            for k, v in mp.items():
                kk = _resolve_metric_name(str(k))
                try:
                    metric_polarity[kk] = float(v)
                except Exception:
                    continue
    if not primary_metrics:
        primary_metrics = list(metrics)
    primary_metrics = [m for m in primary_metrics if m in set(metrics)]
    if not primary_metrics:
        # If prereg primary metrics aren't among the requested metrics, fall back so
        # the report can still compute k-of-n summaries for the requested set.
        primary_metrics = list(metrics)
    if primary_metrics:
        replication_k = int(min(int(replication_k), int(len(primary_metrics))))

    decision_rule = str(args.decision_rule).strip()

    did: Dict[str, Any] = {
        "idle_root": str(idle_root),
        "burn_root": str(burn_root),
        "treatment_source": str(thermal),
        "metrics": metrics,
        "surrogates": surrogates,
        "primary_metrics": primary_metrics,
        "replication_k": int(replication_k),
        "metric_polarity": metric_polarity,
        "decision_rule": decision_rule,
        "by_surrogate": {},
        "by_source_delta": {},
    }

    # Intervention deltas per source.
    sources = sorted({k[0] for k in idle.keys()} & {k[0] for k in burn.keys()})
    for src in sources:
        did["by_source_delta"][src] = {}
        for m in metrics:
            d, used = _paired_diff(idle, burn, src=src, metric=m, sigma=0.0)
            did["by_source_delta"][src][m] = {**_summarize_diff(d, n_boot=int(args.n_boot)), "seeds": used}

    # Within-condition diffs and DiD vs each surrogate.
    for surr in surrogates:
        did["by_surrogate"][surr] = {"within_condition": {}, "did": {}, "decisions": {}}
        for cond_name, rows_map in [("idle", idle), ("burn", burn)]:
            did["by_surrogate"][surr]["within_condition"][cond_name] = {}
            for m in metrics:
                d, used = _paired_diff_between_sources(rows_map, src_a=thermal, src_b=surr, metric=m, sigma=0.0)
                did["by_surrogate"][surr]["within_condition"][cond_name][m] = {**_summarize_diff(d, n_boot=int(args.n_boot)), "seeds": used}

        for m in metrics:
            d_th, used_th = _paired_diff(idle, burn, src=thermal, metric=m, sigma=0.0)
            d_s, used_s = _paired_diff(idle, burn, src=surr, metric=m, sigma=0.0)
            # Align by seed (intersection).
            seed_to_th = {int(seed): float(val) for seed, val in zip(used_th, d_th.tolist())}
            seed_to_s = {int(seed): float(val) for seed, val in zip(used_s, d_s.tolist())}
            common = sorted(set(seed_to_th.keys()) & set(seed_to_s.keys()))
            did_vec = np.asarray([seed_to_th[i] - seed_to_s[i] for i in common], dtype=np.float64)
            did["by_surrogate"][surr]["did"][m] = {**_summarize_diff(did_vec, n_boot=int(args.n_boot)), "seeds": common}

        # Decisions for this surrogate (prereg defaults: k-of-4 in same direction).
        did_dec = _decide_k_of_n_same_direction(
            did["by_surrogate"][surr]["did"],
            metrics=primary_metrics,
            k=int(replication_k),
            polarity=metric_polarity,
        )
        did["by_surrogate"][surr]["decisions"]["did_k_of_n_same_direction"] = did_dec

        within_idle = _decide_k_of_n_ci_excludes_0(
            did["by_surrogate"][surr]["within_condition"]["idle"],
            metrics=primary_metrics,
            k=2,
        )
        within_burn = _decide_k_of_n_ci_excludes_0(
            did["by_surrogate"][surr]["within_condition"]["burn"],
            metrics=primary_metrics,
            k=2,
        )
        did["by_surrogate"][surr]["decisions"]["within_idle_k_of_n_ci_excludes_0"] = within_idle
        did["by_surrogate"][surr]["decisions"]["within_burn_k_of_n_ci_excludes_0"] = within_burn

    # Cross-surrogate within-condition rule (prereg: >=2 surrogates including iaaft).
    iaaft_candidates = [s for s in surrogates if "iaaft" in str(s).lower()]
    within_summary: Dict[str, Any] = {}
    for cond in ["idle", "burn"]:
        passing = [s for s in surrogates if bool(did["by_surrogate"][s]["decisions"][f"within_{cond}_k_of_n_ci_excludes_0"]["pass"])]
        includes_iaaft = bool(set(passing) & set(iaaft_candidates)) if iaaft_candidates else False
        within_summary[cond] = {
            "passing_surrogates": passing,
            "pass_including_iaaft_and_two_surrogates": bool(includes_iaaft and (len(passing) >= 2)),
            "iaaft_candidates": iaaft_candidates,
            "passing_includes_iaaft": includes_iaaft,
        }
    did["within_condition_summary"] = within_summary

    # Cross-surrogate DiD rule (prereg: k-of-n in same direction, and >=2 surrogates incl. iaaft).
    did_summary: Dict[str, Any] = {}
    did_passing = [s for s in surrogates if bool(did["by_surrogate"][s]["decisions"]["did_k_of_n_same_direction"]["pass"])]
    did_includes_iaaft = bool(set(did_passing) & set(iaaft_candidates)) if iaaft_candidates else False
    did_summary["did_k_of_n_same_direction"] = {
        "passing_surrogates": did_passing,
        "pass_including_iaaft_and_two_surrogates": bool(did_includes_iaaft and (len(did_passing) >= 2)),
        "iaaft_candidates": iaaft_candidates,
        "passing_includes_iaaft": did_includes_iaaft,
    }

    # A looser DiD criterion: CI excludes 0 for k-of-n primary metrics (ignores direction).
    for surr in surrogates:
        did["by_surrogate"][surr]["decisions"]["did_k_of_n_ci_excludes_0"] = _decide_k_of_n_ci_excludes_0(
            did["by_surrogate"][surr]["did"],
            metrics=primary_metrics,
            k=int(replication_k),
        )
    did_passing_ci = [s for s in surrogates if bool(did["by_surrogate"][s]["decisions"]["did_k_of_n_ci_excludes_0"]["pass"])]
    did_ci_includes_iaaft = bool(set(did_passing_ci) & set(iaaft_candidates)) if iaaft_candidates else False
    did_summary["did_k_of_n_ci_excludes_0"] = {
        "passing_surrogates": did_passing_ci,
        "pass_including_iaaft_and_two_surrogates": bool(did_ci_includes_iaaft and (len(did_passing_ci) >= 2)),
        "iaaft_candidates": iaaft_candidates,
        "passing_includes_iaaft": did_ci_includes_iaaft,
    }
    did["did_summary"] = did_summary

    # v0.2: PSD-primary PASS/FAIL rule (seed-paired DiD vs surrogates).
    decision: Dict[str, Any] = {"decision": "UNSPECIFIED", "rule": str(decision_rule)}
    if decision_rule == "v0_2_psd_primary":
        per_surrogate: Dict[str, Any] = {}
        for surr in surrogates:
            per_surrogate[surr] = _decide_k_of_n_matches_polarity(
                did["by_surrogate"][surr]["did"],
                metrics=primary_metrics,
                k=int(replication_k),
                polarity=metric_polarity,
            )

        iaaft_candidates = [s for s in surrogates if "iaaft" in str(s).lower()]
        other_candidates = [s for s in surrogates if s not in iaaft_candidates]
        passing = [s for s in surrogates if bool(per_surrogate.get(s, {}).get("pass"))]
        passing_iaaft = [s for s in passing if s in iaaft_candidates]
        passing_other = [s for s in passing if s in other_candidates]
        overall_pass = bool((len(passing_iaaft) >= 1) and (len(passing_other) >= 2))

        decision = {
            "decision": "PASS" if overall_pass else "FAIL",
            "rule": "v0_2_psd_primary",
            "replication_k": int(replication_k),
            "primary_metrics": list(primary_metrics),
            "per_surrogate": per_surrogate,
            "passing_surrogates": passing,
            "passing_iaaft": passing_iaaft,
            "passing_non_iaaft": passing_other,
            "requirement": {"needs_iaaft": True, "needs_non_iaaft_count": 2},
        }

        # Confound rejection gates (best-effort) when burn condition is batching/jitter.
        burn_proto = None
        if (burn_root / "protocol.json").exists():
            try:
                burn_proto = json.loads((burn_root / "protocol.json").read_text(encoding="utf-8"))
            except Exception:
                burn_proto = None
        burn_prereg = None
        if (burn_root / "preregistered_metrics.json").exists():
            try:
                burn_prereg = json.loads((burn_root / "preregistered_metrics.json").read_text(encoding="utf-8"))
            except Exception:
                burn_prereg = None
        burn_condition = burn_prereg.get("condition") if isinstance(burn_prereg, dict) else None
        if str(burn_condition) in {"batching", "jitter"}:
            tol = float(dict(dict(burn_prereg.get("decision_thresholds", {})).get("confound_rejection", {})).get("target_tol_hz", 0.15)) if isinstance(burn_prereg, dict) else 0.15
            vals = []
            for (src, _seed, sg), d in burn.items():
                if src == thermal and float(sg) == 0.0:
                    vals.append(float(d.get("psd_peak_hz_error_hz_mean", 0.0)))
            vals = [v for v in vals if np.isfinite(v)]
            err_med = float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")
            c0_pass = bool(np.isfinite(err_med) and (err_med > tol))
            conf: Dict[str, Any] = {
                "condition": str(burn_condition),
                "target_tol_hz": float(tol),
                "psd_peak_hz_error_hz_mean_median": float(err_med),
                "C0_no_target_lock_pass": bool(c0_pass),
            }
            if str(burn_condition) == "batching" and isinstance(burn_proto, dict):
                try:
                    flush_ms = float(dict(dict(burn_proto.get("pow", {})).get("confound", {})).get("batch_flush_ms"))
                except Exception:
                    flush_ms = float("nan")
                if np.isfinite(flush_ms) and flush_ms > 0:
                    expected = 1000.0 / flush_ms
                    # Pull psd_peak_hz_median directly from the burn SUMMARY.csv so callers don't need
                    # to include it in --metrics.
                    hz_vals = []
                    for r in burn_rows:
                        try:
                            if str(r.get("source")) != str(thermal):
                                continue
                            if float(r.get("sigma", 0.0) or 0.0) != 0.0:
                                continue
                            hz_vals.append(float(r.get("psd_peak_hz_median", 0.0) or 0.0))
                        except Exception:
                            continue
                    hz_vals = [v for v in hz_vals if np.isfinite(v)]
                    hz_med = float(np.median(np.asarray(hz_vals, dtype=np.float64))) if hz_vals else float("nan")
                    conf["batching_expected_hz"] = float(expected)
                    conf["psd_peak_hz_median_median"] = float(hz_med)
                    conf["psd_peak_hz_error_to_batching_hz_median"] = float(abs(hz_med - expected)) if np.isfinite(hz_med) else float("nan")
            decision["confound_gates"] = conf

    did["decision"] = decision

    # DiD vs baselines (PRNG and timing).
    for baseline in ["prng", "timing_jitter"]:
        if baseline not in sources:
            continue
        key = f"did_vs_{baseline}"
        did[key] = {}
        for m in metrics:
            d_th, used_th = _paired_diff(idle, burn, src=thermal, metric=m, sigma=0.0)
            d_b, used_b = _paired_diff(idle, burn, src=baseline, metric=m, sigma=0.0)
            seed_to_th = {int(seed): float(val) for seed, val in zip(used_th, d_th.tolist())}
            seed_to_b = {int(seed): float(val) for seed, val in zip(used_b, d_b.tolist())}
            common = sorted(set(seed_to_th.keys()) & set(seed_to_b.keys()))
            did_vec = np.asarray([seed_to_th[i] - seed_to_b[i] for i in common], dtype=np.float64)
            did[key][m] = {**_summarize_diff(did_vec, n_boot=int(args.n_boot)), "seeds": common}

    payload = {"preregistration": prereg, **did}
    _write_json(out / "REPORT.json", payload)
    _write_json(out / "results.json", payload)

    # Markdown report.
    md: List[str] = []
    md.append("# Difference-in-Differences (seed paired)")
    md.append("")
    md.append(f"- Idle: `{idle_root}`")
    md.append(f"- Burn: `{burn_root}`")
    md.append(f"- Metrics: `{','.join(metrics)}`")
    md.append(f"- Surrogates: `{','.join(surrogates)}`")
    md.append("")

    if prereg:
        md.append("## Preregistration")
        md.append("")
        md.append(f"- primary_metrics: `{','.join([str(x) for x in prereg.get('primary_metrics', [])])}`")
        md.append(f"- primary_effect_threshold: `{prereg.get('primary_effect_threshold')}`")
        rr = prereg.get("replication_rule", {})
        if isinstance(rr, dict):
            md.append(f"- replication_rule: k={rr.get('same_direction_at_least_k')} of n={rr.get('out_of_n')}")
        md.append("")

    md.append("## Decision Summary")
    md.append("")
    md.append("| category | condition | passing_surrogates | pass_requires_iaaft_and_2_surrogates |")
    md.append("|---|---|---|---:|")
    for cond in ["idle", "burn"]:
        ps = ", ".join(within_summary.get(cond, {}).get("passing_surrogates", []))
        ok = bool(within_summary.get(cond, {}).get("pass_including_iaaft_and_two_surrogates", False))
        md.append(f"| within | {cond} | {ps} | {str(ok)} |")
    md.append("| did | k-of-n same-direction | " + ", ".join(did_summary["did_k_of_n_same_direction"]["passing_surrogates"]) + f" | {str(bool(did_summary['did_k_of_n_same_direction']['pass_including_iaaft_and_two_surrogates']))} |")
    md.append("| did | k-of-n CI excludes 0 | " + ", ".join(did_summary["did_k_of_n_ci_excludes_0"]["passing_surrogates"]) + f" | {str(bool(did_summary['did_k_of_n_ci_excludes_0']['pass_including_iaaft_and_two_surrogates']))} |")
    if isinstance(did.get("decision"), dict) and str(did["decision"].get("decision", "")).strip():
        md.append("| v0.2 | overall | " + ", ".join(did["decision"].get("passing_surrogates", [])) + f" | {str(did['decision']['decision'] == 'PASS')} |")
    md.append("")

    conf = dict(did.get("decision", {}).get("confound_gates", {}) or {}) if isinstance(did.get("decision"), dict) else {}
    if conf:
        md.append("## Confound Gates (v0.2)")
        md.append("")
        md.append(f"- condition: `{conf.get('condition')}`")
        md.append(f"- C0_no_target_lock_pass: `{conf.get('C0_no_target_lock_pass')}` (tol_hz={conf.get('target_tol_hz')}, median_err={conf.get('psd_peak_hz_error_hz_mean_median')})")
        if conf.get("condition") == "batching":
            md.append(f"- batching_expected_hz: `{conf.get('batching_expected_hz')}`")
            md.append(f"- psd_peak_hz_median_median: `{conf.get('psd_peak_hz_median_median')}`")
            md.append(f"- psd_peak_hz_error_to_batching_hz_median: `{conf.get('psd_peak_hz_error_to_batching_hz_median')}`")
        md.append("")

    def _fmt(x: Dict[str, Any]) -> str:
        return f"{x['mean']:.6g} [{x['ci95_lo']:.6g}, {x['ci95_hi']:.6g}] (d={x['cohens_d_paired']:.3g})"

    for surr in surrogates:
        md.append(f"## Surrogate: `{surr}`")
        md.append("")
        dec = did["by_surrogate"][surr]["decisions"]["did_k_of_n_same_direction"]
        md.append(
            f"- DiD prereg check (k-of-n, same direction): pass={dec['pass']} "
            f"(k={dec['k']} of n={dec['n']}, direction={dec['direction']}, passed={','.join(dec['passed'])})"
        )
        md.append("")
        md.append("### Within-condition diffs (thermal − surrogate)")
        md.append("")
        md.append("| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |")
        md.append("|---|---:|---:|")
        for m in metrics:
            wi = did["by_surrogate"][surr]["within_condition"]["idle"][m]
            wb = did["by_surrogate"][surr]["within_condition"]["burn"][m]
            md.append(f"| {m} | {_fmt(wi)} | {_fmt(wb)} |")
        md.append("")

        md.append("### DiD (Δthermal − Δsurrogate)")
        md.append("")
        md.append("| metric | did mean (ci95) | ci_excludes_0 |")
        md.append("|---|---:|---:|")
        pass_count = 0
        for m in metrics:
            d = did["by_surrogate"][surr]["did"][m]
            ce = bool(d["ci_excludes_0"])
            pass_count += int(ce)
            md.append(f"| {m} | {_fmt(d)} | {str(ce)} |")
        md.append("")
        md.append(f"- DiD metrics with CI excluding 0: {pass_count}/{len(metrics)}")
        md.append("")

    if isinstance(did.get("decision"), dict) and str(did["decision"].get("decision", "")).strip():
        md.append(f"DECISION: {did['decision']['decision']}")
        md.append("")

    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_manifest_sha256(out / "MANIFEST.sha256", files=list_files_for_manifest(out))
    print("wrote:", str(out))


if __name__ == "__main__":
    main()
