from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _popcount_bytes(x: bytes) -> int:
    # Python 3.10 int.bit_count is fast enough here.
    return int(int.from_bytes(x, "big", signed=False).bit_count())


def _hamming_norm_hex(a_hex: str, b_hex: str) -> float:
    a = bytes.fromhex(str(a_hex))
    b = bytes.fromhex(str(b_hex))
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    x = bytes([aa ^ bb for aa, bb in zip(a, b)])
    return float(_popcount_bytes(x)) / float(8 * len(a))


def validate_events_timestamps(t_ns: np.ndarray) -> None:
    t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    if t.size < 2:
        return
    if not np.all(np.isfinite(t.astype(np.float64))):
        raise ValueError("events timestamps contain non-finite values")
    if not np.all(t[1:] > t[:-1]):
        raise ValueError("events.t_ns must be strictly increasing")


def _regular_binned_count_series(t_ns: np.ndarray, *, bin_dt_s: float) -> Tuple[np.ndarray, float]:
    t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    if t.size == 0:
        return np.zeros((0,), dtype=np.float64), float(bin_dt_s)
    dt = float(bin_dt_s)
    if not (dt > 0):
        dt = 0.05
    t0 = int(t[0])
    # bins over [t0, t_last]
    span_s = float((int(t[-1]) - t0) * 1e-9)
    n_bins = int(max(1, math.ceil(span_s / dt)))
    idx = np.floor(((t - t0).astype(np.float64) * 1e-9) / dt).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return counts, float(dt)


def psd_peak_features(counts: np.ndarray, *, bin_dt_s: float, f_min_hz: float = 0.1, f_max_hz: float = 20.0) -> Dict[str, float]:
    x = np.asarray(counts, dtype=np.float64).reshape(-1)
    dt = float(bin_dt_s)
    if x.size < 8 or not (dt > 0):
        return {
            "psd_peak_hz": 0.0,
            "psd_peak_power": 0.0,
            "psd_peak_snr_db": 0.0,
            "psd_peak_bandwidth_hz": 0.0,
            "psd_peak_q": 0.0,
        }
    y = x - float(np.mean(x))
    w = np.hanning(int(y.size))
    yw = y * w
    Y = np.fft.rfft(yw)
    P = (np.abs(Y) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(int(y.size), d=float(dt))
    # Exclude DC and restrict frequency band.
    mask = (freqs >= float(f_min_hz)) & (freqs <= float(f_max_hz))
    if not np.any(mask):
        return {
            "psd_peak_hz": 0.0,
            "psd_peak_power": 0.0,
            "psd_peak_snr_db": 0.0,
            "psd_peak_bandwidth_hz": 0.0,
            "psd_peak_q": 0.0,
        }
    Pm = P[mask]
    fm = freqs[mask]
    j = int(np.argmax(Pm))
    peak_power = float(Pm[j])
    peak_hz = float(fm[j])
    floor = float(np.median(Pm)) if Pm.size else 0.0
    snr = 10.0 * math.log10((peak_power + 1e-12) / (floor + 1e-12)) if peak_power > 0 else 0.0
    # Q factor estimate: bw = full-width at half-power (≈ -3 dB).
    bw = 0.0
    q = 0.0
    if peak_power > 0 and Pm.size >= 3:
        half = peak_power * 0.5
        # Find contiguous region around the peak where P >= half.
        left = j
        while left > 0 and float(Pm[left - 1]) >= half:
            left -= 1
        right = j
        while right + 1 < int(Pm.size) and float(Pm[right + 1]) >= half:
            right += 1
        if right > left:
            bw = float(fm[right] - fm[left])
        # If the peak is a single bin, treat bw as one-bin width (frequency resolution).
        if not (bw > 0):
            if fm.size >= 2:
                bw = float(abs(fm[min(int(fm.size - 1), j + 1)] - fm[max(0, j - 1)]))
            else:
                bw = 0.0
        if bw > 0:
            q = float(peak_hz / bw)
    return {
        "psd_peak_hz": float(peak_hz),
        "psd_peak_power": float(peak_power),
        "psd_peak_snr_db": float(snr),
        "psd_peak_bandwidth_hz": float(bw),
        "psd_peak_q": float(q),
    }


def _entropy_hist(x: np.ndarray, *, bin_edges: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if edges.size < 3:
        return 0.0
    h, _ = np.histogram(y, bins=edges, density=False)
    p = h.astype(np.float64)
    p = p / max(1.0, float(p.sum()))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    ent = -float(np.sum(p * np.log(p)))
    # Normalize by log(#bins) to [0,1] (approx).
    ent = ent / float(np.log(max(2, int(edges.size - 1))))
    return float(ent)


@dataclass(frozen=True)
class ChronosMetricsConfig:
    schema_version: str = "0.1"
    window_events: int = 64
    psd_bin_dt_s: float = 0.05
    entropy_bins: int = 32
    log_delta_clip: float = 12.0  # clip log10(delta_s) to [-clip, clip]


def derive_deltas_csv(events_csv: Path, *, out_path: Path) -> Dict[str, Any]:
    rows = _read_csv_dicts(Path(events_csv))
    t = np.asarray([int(r["t_ns"]) for r in rows], dtype=np.int64)
    validate_events_timestamps(t)
    if t.size < 2:
        out_path.write_text("t_ns,delta_s\n", encoding="utf-8")
        return {"n_events": int(t.size), "n_deltas": 0}
    d = (t[1:] - t[:-1]).astype(np.float64) * 1e-9
    if np.any(d <= 0):
        raise ValueError("delta_s must be positive")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_ns", "delta_s"])
        w.writeheader()
        for i in range(int(d.size)):
            w.writerow({"t_ns": int(t[i + 1]), "delta_s": float(d[i])})
    return {"n_events": int(t.size), "n_deltas": int(d.size)}


def fit_entropy_bin_edges_from_deltas(deltas_s: np.ndarray, *, n_bins: int, log_delta_clip: float) -> np.ndarray:
    d = np.asarray(deltas_s, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return np.linspace(-1.0, 1.0, int(max(3, n_bins + 1)), dtype=np.float64)
    y = np.log10(np.maximum(d, 1e-12))
    clip = float(max(1.0, log_delta_clip))
    y = np.clip(y, -clip, clip)
    lo = float(np.quantile(y, 0.01))
    hi = float(np.quantile(y, 0.99))
    if not (hi > lo):
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(max(3, n_bins + 1)), dtype=np.float64)
    return edges


def compute_chronos_metrics_from_events(
    events_csv: Path,
    *,
    out_csv: Path,
    out_meta: Path,
    cfg: ChronosMetricsConfig,
    entropy_bin_edges: Optional[np.ndarray] = None,
    expected_heartbeat_hz: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = ChronosMetricsConfig(**{**cfg.__dict__})
    rows = _read_csv_dicts(Path(events_csv))
    t = np.asarray([int(r["t_ns"]) for r in rows], dtype=np.int64)
    validate_events_timestamps(t)
    hashes = [str(r.get("hash_hex", "")) for r in rows]

    if t.size < 2:
        out_csv.write_text("", encoding="utf-8")
        _write_json(out_meta, {"schema_version": cfg.schema_version, "error": "insufficient_events"})
        return {"n_events": int(t.size), "n_windows": 0}

    deltas_s = (t[1:] - t[:-1]).astype(np.float64) * 1e-9
    if np.any(deltas_s <= 0):
        raise ValueError("delta_s must be positive")

    edges = np.asarray(entropy_bin_edges, dtype=np.float64).reshape(-1) if entropy_bin_edges is not None else fit_entropy_bin_edges_from_deltas(deltas_s, n_bins=int(cfg.entropy_bins), log_delta_clip=float(cfg.log_delta_clip))

    W = int(cfg.window_events)
    if W < 4:
        raise ValueError("window_events must be >= 4")
    n_win = int(max(0, int(t.size) - W + 1))
    out_rows: List[Dict[str, Any]] = []
    for i in range(n_win):
        t0 = int(t[i])
        t1 = int(t[i + W - 1])
        # Window deltas are between events, length W-1.
        ds = deltas_s[i : i + W - 1]
        cv = float(np.std(ds, ddof=1) / max(1e-12, float(np.mean(ds)))) if ds.size >= 2 else 0.0
        y = np.log10(np.maximum(ds, 1e-12))
        y = np.clip(y, -float(cfg.log_delta_clip), float(cfg.log_delta_clip))
        ent = _entropy_hist(y, bin_edges=edges)

        # Hamming on hashes in this event window.
        ham = 0.0
        if W >= 2:
            hs = hashes[i : i + W]
            if len(hs) >= 2:
                ham = float(np.mean([_hamming_norm_hex(hs[j], hs[j + 1]) for j in range(len(hs) - 1)]))

        # PSD on binned event-count series within this event window.
        counts, dt = _regular_binned_count_series(t[i : i + W], bin_dt_s=float(cfg.psd_bin_dt_s))
        psd = psd_peak_features(counts, bin_dt_s=float(dt))
        peak_hz = float(psd["psd_peak_hz"])
        hz_err = float(abs(peak_hz - float(expected_heartbeat_hz))) if expected_heartbeat_hz is not None else float("nan")
        span_s = float((t1 - t0) * 1e-9) if t1 > t0 else 0.0
        rate = float(W / max(1e-12, span_s)) if span_s > 0 else 0.0

        out_rows.append(
            {
                "window_idx": int(i),
                "window_t0_ns": int(t0),
                "window_t1_ns": int(t1),
                "cv": float(cv),
                "hist_entropy": float(ent),
                "hamming_norm": float(ham),
                "event_rate_hz": float(rate),
                "psd_peak_hz": float(peak_hz),
                "psd_peak_power": float(psd["psd_peak_power"]),
                "psd_peak_snr_db": float(psd["psd_peak_snr_db"]),
                "psd_peak_bandwidth_hz": float(psd.get("psd_peak_bandwidth_hz", 0.0)),
                "psd_peak_q": float(psd.get("psd_peak_q", 0.0)),
                "psd_peak_hz_error_hz": float(hz_err),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "window_idx",
                "window_t0_ns",
                "window_t1_ns",
                "cv",
                "hist_entropy",
                "hamming_norm",
                "event_rate_hz",
                "psd_peak_hz",
                "psd_peak_power",
                "psd_peak_snr_db",
                "psd_peak_bandwidth_hz",
                "psd_peak_q",
                "psd_peak_hz_error_hz",
            ],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    meta = {
        "schema_version": str(cfg.schema_version),
        "window_events": int(W),
        "psd_bin_dt_s": float(cfg.psd_bin_dt_s),
        "entropy_bins": int(cfg.entropy_bins),
        "entropy_bin_edges": edges.astype(np.float64).tolist(),
        "log_delta_clip": float(cfg.log_delta_clip),
        "expected_heartbeat_hz": float(expected_heartbeat_hz) if expected_heartbeat_hz is not None else None,
        "counts": {"n_events": int(t.size), "n_windows": int(n_win)},
    }
    _write_json(out_meta, meta)
    return {"n_events": int(t.size), "n_windows": int(n_win)}


def summarize_chronos_metrics(metrics_csv: Path) -> Dict[str, Any]:
    rows = _read_csv_dicts(Path(metrics_csv))
    if not rows:
        return {
            "cv_mean": 0.0,
            "hist_entropy_mean": 0.0,
            "psd_peak_snr_db_mean": 0.0,
            "psd_peak_hz_median": 0.0,
            "psd_peak_hz_error_hz_mean": 0.0,
            "psd_peak_q_mean": 0.0,
            "psd_peak_hz_iqr_hz": 0.0,
        }

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, 0.0) or 0.0) for r in rows], dtype=np.float64)

    cv = col("cv")
    ent = col("hist_entropy")
    snr = col("psd_peak_snr_db")
    peak_hz = col("psd_peak_hz")
    hz_err = col("psd_peak_hz_error_hz")
    hz_err = hz_err[np.isfinite(hz_err)]
    q = col("psd_peak_q")

    return {
        "cv_mean": float(cv.mean()),
        "hist_entropy_mean": float(ent.mean()),
        "psd_peak_snr_db_mean": float(snr.mean()),
        "psd_peak_hz_median": float(np.median(peak_hz)),
        "psd_peak_hz_error_hz_mean": float(hz_err.mean()) if hz_err.size else 0.0,
        "psd_peak_q_mean": float(q.mean()),
        "psd_peak_hz_iqr_hz": float(np.quantile(peak_hz, 0.75) - np.quantile(peak_hz, 0.25)) if peak_hz.size else 0.0,
    }
