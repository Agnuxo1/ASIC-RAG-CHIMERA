from __future__ import annotations

import json
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from chimera_handoff.entropy.sources import EntropySource


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    lag = int(lag)
    if x.size <= lag or lag <= 0:
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def _psd_slope(x: np.ndarray) -> float:
    """
    Very coarse “spectral color” proxy: slope of log PSD vs log f.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 64:
        return 0.0
    x = x - x.mean()
    y = np.fft.rfft(x)
    p = (y.real * y.real + y.imag * y.imag)
    f = np.fft.rfftfreq(x.size, d=1.0)
    # Drop DC and the last bin (can be unstable at Nyquist).
    f = f[1:-1]
    p = p[1:-1]
    m = (f > 0) & np.isfinite(p) & (p > 0)
    f = f[m]
    p = p[m]
    if f.size < 8:
        return 0.0
    xf = np.log(f)
    yp = np.log(p)
    A = np.vstack([xf, np.ones_like(xf)]).T
    slope, _ = np.linalg.lstsq(A, yp, rcond=None)[0]
    return float(slope)


def profile_entropy_source(
    source: object,
    *,
    duration_sec: float = 2.0,
    block_bytes: int = 65536,
    max_bytes: int = 2 * 1024 * 1024,
    hist_bins: int = 32,
    autocorr_lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
) -> Dict[str, Any]:
    # Float-stream sources (e.g. thermal_residual) are profiled differently.
    if hasattr(source, "read_floats"):
        return profile_float_stream(  # type: ignore[arg-type]
            source,
            duration_sec=float(duration_sec),
            block_floats=max(16, int(block_bytes) // 4),
            max_floats=max(256, int(max_bytes) // 4),
            hist_bins=int(hist_bins),
            autocorr_lags=autocorr_lags,
        )

    if not isinstance(source, EntropySource):
        # Best-effort: allow duck-typed byte sources.
        source = source  # type: ignore[no-redef]

    duration_sec = float(duration_sec)
    block_bytes = int(block_bytes)
    hist_bins = int(hist_bins)

    t0 = time.perf_counter()
    samples: List[bytes] = []
    lat: List[float] = []
    total = 0
    while (time.perf_counter() - t0) < duration_sec:
        if total >= int(max_bytes):
            break
        s0 = time.perf_counter()
        b = source.read_bytes(block_bytes)
        s1 = time.perf_counter()
        lat.append(float(s1 - s0))
        if not b:
            break
        samples.append(b)
        total += len(b)

    dt = float(time.perf_counter() - t0)
    raw = b"".join(samples)
    arr = np.frombuffer(raw, dtype=np.uint8)

    hist, edges = np.histogram(arr.astype(np.float64), bins=hist_bins, range=(0.0, 256.0), density=True)
    mean = float(arr.mean()) if arr.size else 0.0
    var = float(arr.var()) if arr.size else 0.0

    comp = zlib.compress(raw) if raw else b""
    comp_ratio = float(len(comp) / max(1, len(raw)))

    x = arr.astype(np.float64)
    ac = {str(l): _autocorr(x, int(l)) for l in autocorr_lags}
    slope = _psd_slope(x)

    # Stationarity proxy: first half vs second half.
    mid = int(arr.size // 2)
    a = arr[:mid]
    b = arr[mid:]
    stationarity = {
        "first_half_mean": float(a.mean()) if a.size else 0.0,
        "second_half_mean": float(b.mean()) if b.size else 0.0,
        "first_half_var": float(a.var()) if a.size else 0.0,
        "second_half_var": float(b.var()) if b.size else 0.0,
    }

    lat = np.asarray(lat, dtype=np.float64)
    burst = {
        "n_reads": int(lat.size),
        "latency_s_mean": float(lat.mean()) if lat.size else 0.0,
        "latency_s_std": float(lat.std(ddof=1)) if lat.size >= 2 else 0.0,
        "latency_s_p95": float(np.quantile(lat, 0.95)) if lat.size else 0.0,
    }

    return {
        "source_id": getattr(source, "id", "unknown"),
        "duration_sec": float(duration_sec),
        "block_bytes": int(block_bytes),
        "max_bytes": int(max_bytes),
        "bytes_total": int(total),
        "throughput_bytes_per_sec": float(total / dt) if dt > 0 else 0.0,
        "hist": {"bins": int(hist_bins), "edges": edges.tolist(), "density": hist.tolist()},
        "mean": float(mean),
        "variance": float(var),
        "compressibility_ratio_zlib": float(comp_ratio),
        "autocorr": ac,
        "psd_loglog_slope": float(slope),
        "burstiness": burst,
        "stationarity_proxy": stationarity,
    }


def profile_float_stream(
    source: object,
    *,
    duration_sec: float = 2.0,
    block_floats: int = 16384,
    max_floats: int = 512 * 1024,
    hist_bins: int = 32,
    autocorr_lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
) -> Dict[str, Any]:
    duration_sec = float(duration_sec)
    block_floats = int(block_floats)
    max_floats = int(max_floats)
    hist_bins = int(hist_bins)

    read_floats = getattr(source, "read_floats", None)
    if not callable(read_floats):
        raise TypeError("source does not support read_floats")

    t0 = time.perf_counter()
    blocks: List[np.ndarray] = []
    lat: List[float] = []
    total = 0
    while (time.perf_counter() - t0) < duration_sec:
        if total >= int(max_floats):
            break
        s0 = time.perf_counter()
        x = np.asarray(read_floats(int(block_floats)), dtype=np.float32).reshape(-1)
        s1 = time.perf_counter()
        lat.append(float(s1 - s0))
        if x.size == 0:
            break
        blocks.append(x)
        total += int(x.size)

    dt = float(time.perf_counter() - t0)
    arr = np.concatenate(blocks, axis=0) if blocks else np.zeros((0,), dtype=np.float32)
    x = arr.astype(np.float64, copy=False)

    hist, edges = np.histogram(x, bins=int(hist_bins), range=(float(np.min(x)) if x.size else -1.0, float(np.max(x)) if x.size else 1.0), density=True)
    mean = float(x.mean()) if x.size else 0.0
    var = float(x.var()) if x.size else 0.0

    ac = {str(l): _autocorr(x, int(l)) for l in autocorr_lags}
    slope = _psd_slope(x)

    mid = int(x.size // 2)
    a = x[:mid]
    b = x[mid:]
    stationarity = {
        "first_half_mean": float(a.mean()) if a.size else 0.0,
        "second_half_mean": float(b.mean()) if b.size else 0.0,
        "first_half_var": float(a.var()) if a.size else 0.0,
        "second_half_var": float(b.var()) if b.size else 0.0,
    }

    lat = np.asarray(lat, dtype=np.float64)
    burst = {
        "n_reads": int(lat.size),
        "latency_s_mean": float(lat.mean()) if lat.size else 0.0,
        "latency_s_std": float(lat.std(ddof=1)) if lat.size >= 2 else 0.0,
        "latency_s_p95": float(np.quantile(lat, 0.95)) if lat.size else 0.0,
    }

    return {
        "source_id": getattr(source, "id", "unknown"),
        "stream_kind": "float32",
        "duration_sec": float(duration_sec),
        "block_floats": int(block_floats),
        "max_floats": int(max_floats),
        "floats_total": int(total),
        "throughput_samples_per_sec": float(total / dt) if dt > 0 else 0.0,
        "hist": {"bins": int(hist_bins), "edges": edges.tolist(), "density": hist.tolist()},
        "mean": float(mean),
        "variance": float(var),
        "autocorr": ac,
        "psd_loglog_slope": float(slope),
        "burstiness": burst,
        "stationarity_proxy": stationarity,
    }


def write_entropy_profile(path: Path, profile: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
