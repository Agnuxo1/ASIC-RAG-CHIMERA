from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class HealthCheckResult:
    ok: bool
    degraded: bool
    failures: List[str]
    details: Dict[str, Any]


def _max_run_length_bytes(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    # Run-length encoding max.
    x = np.asarray(arr, dtype=np.uint8).reshape(-1)
    max_run = 1
    run = 1
    for i in range(1, int(x.size)):
        if int(x[i]) == int(x[i - 1]):
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    return int(max_run)


def _adaptive_proportion_bits(bits: np.ndarray, *, w: int, lo: int, hi: int) -> Tuple[int, int, int]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size < w:
        return 0, 0, int(bits.size)
    fail = 0
    total = 0
    ones = int(bits.sum())
    # Sliding window via cumulative sum for speed.
    c = np.concatenate([[0], np.cumsum(bits, dtype=np.int64)])
    for i in range(0, int(bits.size) - int(w) + 1, int(w)):
        s = int(c[i + w] - c[i])
        total += 1
        if s < int(lo) or s > int(hi):
            fail += 1
    return int(fail), int(total), int(ones)


def run_health_tests(
    sample: Union[bytes, bytearray, np.ndarray],
    *,
    rep_max: int = 32,
    stuck_frac: float = 0.95,
    apt_window_bits: int = 1024,
    apt_lo: int = 256,
    apt_hi: int = 768,
) -> HealthCheckResult:
    """
    Lightweight gross-failure detectors inspired by NIST SP 800-90B health test concepts.
    These are *sanity checks*, not compliance/certification.
    """
    if isinstance(sample, (bytes, bytearray)):
        return _run_health_tests_bytes(
            bytes(sample),
            rep_max=int(rep_max),
            stuck_frac=float(stuck_frac),
            apt_window_bits=int(apt_window_bits),
            apt_lo=int(apt_lo),
            apt_hi=int(apt_hi),
        )
    return _run_health_tests_floats(
        np.asarray(sample, dtype=np.float64).reshape(-1),
        rep_max=int(rep_max),
        stuck_frac=float(stuck_frac),
        apt_window_bits=int(apt_window_bits),
        apt_lo=int(apt_lo),
        apt_hi=int(apt_hi),
    )


def _run_health_tests_bytes(
    sample_bytes: bytes,
    *,
    rep_max: int,
    stuck_frac: float,
    apt_window_bits: int,
    apt_lo: int,
    apt_hi: int,
) -> HealthCheckResult:
    arr = np.frombuffer(sample_bytes, dtype=np.uint8)
    failures: List[str] = []
    details: Dict[str, Any] = {}

    # Repetition count test on bytes.
    max_run = _max_run_length_bytes(arr)
    details["repetition_count"] = {"max_run_len_bytes": int(max_run), "threshold": int(rep_max)}
    if max_run >= int(rep_max):
        failures.append("repetition_count_test_failed")

    # Stuck-byte detector.
    if arr.size:
        counts = np.bincount(arr.astype(np.int64), minlength=256)
        frac = float(counts.max() / float(arr.size))
    else:
        frac = 0.0
    details["stuck_byte"] = {"max_symbol_frac": float(frac), "threshold": float(stuck_frac)}
    if frac >= float(stuck_frac):
        failures.append("stuck_byte_failed")

    # Adaptive proportion test on bits (very coarse).
    bits = np.unpackbits(arr, bitorder="little") if arr.size else np.zeros((0,), dtype=np.uint8)
    fail, total, ones = _adaptive_proportion_bits(bits, w=int(apt_window_bits), lo=int(apt_lo), hi=int(apt_hi))
    details["adaptive_proportion_bits"] = {"fail_windows": int(fail), "total_windows": int(total), "ones": int(ones)}
    if total > 0 and fail > 0:
        failures.append("adaptive_proportion_test_failed")

    ok = len(failures) == 0
    # Policy: any gross-failure is degraded.
    degraded = not ok
    return HealthCheckResult(ok=bool(ok), degraded=bool(degraded), failures=failures, details=details)


def _run_health_tests_floats(
    x: np.ndarray,
    *,
    rep_max: int,
    stuck_frac: float,
    apt_window_bits: int,
    apt_lo: int,
    apt_hi: int,
) -> HealthCheckResult:
    failures: List[str] = []
    details: Dict[str, Any] = {"stream_kind": "float"}

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    details["n"] = int(x.size)
    if x.size == 0:
        failures.append("empty_stream")
        return HealthCheckResult(ok=False, degraded=True, failures=failures, details=details)
    if not np.all(np.isfinite(x)):
        failures.append("non_finite_values")

    mean = float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else 0.0
    std = float(np.std(x[np.isfinite(x)], ddof=1)) if np.isfinite(x).sum() >= 2 else 0.0
    details["mean"] = float(mean)
    details["std"] = float(std)
    if not (std > 1e-9):
        failures.append("near_constant_stream")

    # Coarse “sign dynamics” diagnostics only; do not treat non-uniformity as degraded.
    sym = np.zeros((x.size,), dtype=np.int8)
    sym[x > 0] = 1
    sym[x < 0] = -1
    transitions = int(np.sum(sym[1:] != sym[:-1])) if sym.size >= 2 else 0
    details["sign_summary"] = {
        "frac_pos": float(np.mean(sym == 1)),
        "frac_zero": float(np.mean(sym == 0)),
        "frac_neg": float(np.mean(sym == -1)),
        "transition_rate": float(transitions / max(1, int(sym.size - 1))),
    }

    ok = len(failures) == 0
    degraded = not ok
    return HealthCheckResult(ok=bool(ok), degraded=bool(degraded), failures=failures, details=details)
