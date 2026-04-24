from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.thermal_telemetry import ThermalSample


@dataclass(frozen=True)
class RCModelParams:
    a: float
    b: float
    c: float
    u_kind: str  # "power_w" | "util_freq" | "util" | "zero"
    fit_n: int


def _u_from_samples(samples: List[ThermalSample]) -> Tuple[np.ndarray, str]:
    util = np.asarray([float(s.util) for s in samples], dtype=np.float64)
    freq = np.asarray([float(s.freq_mhz) for s in samples], dtype=np.float64)
    power = np.asarray([np.nan if s.power_w is None else float(s.power_w) for s in samples], dtype=np.float64)

    if np.isfinite(power).sum() >= max(4, int(len(samples) // 4)):
        u = np.where(np.isfinite(power), power, np.nanmedian(power))
        return u.astype(np.float64), "power_w"

    if np.isfinite(freq).sum() >= max(4, int(len(samples) // 4)):
        u = util * np.where(np.isfinite(freq), freq, 0.0)
        return u.astype(np.float64), "util_freq"

    if np.isfinite(util).sum() >= 2:
        return util.astype(np.float64), "util"

    return np.zeros((len(samples),), dtype=np.float64), "zero"


def fit_first_order_rc(
    samples: List[ThermalSample],
    *,
    calib_n: int,
    clamp_a: Tuple[float, float] = (1e-6, 0.999999),
) -> Tuple[RCModelParams, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit y_{t+1} ≈ a y_t + b u_t + c by least squares on a calibration prefix.
    Then simulate forward to get yhat and residuals r = y - yhat.
    """
    if len(samples) < 4:
        raise ValueError("need at least 4 samples to fit RC model")

    y = np.asarray([float(s.temp_c) for s in samples], dtype=np.float64)
    u, u_kind = _u_from_samples(samples)

    n = int(len(samples))
    calib_n = int(min(max(3, int(calib_n)), n - 1))

    # Fit on (t=0..calib_n-2): predict y_{t+1} from (y_t, u_t, 1).
    yt = y[: calib_n - 1]
    ut = u[: calib_n - 1]
    y_next = y[1:calib_n]
    X = np.stack([yt, ut, np.ones_like(yt)], axis=1)
    beta, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    a0, b0, c0 = [float(x) for x in beta.tolist()]

    # Enforce stable constraints.
    lo, hi = float(clamp_a[0]), float(clamp_a[1])
    a = float(np.clip(a0, lo, hi))
    b = float(max(0.0, b0))
    c = float(c0)

    yhat = np.empty((n,), dtype=np.float64)
    yhat[0] = float(y[0])
    for t in range(int(n - 1)):
        yhat[t + 1] = a * yhat[t] + b * float(u[t]) + c

    r = y - yhat
    meta: Dict[str, Any] = {
        "fit": {"calib_n": int(calib_n), "a_raw": float(a0), "b_raw": float(b0), "c_raw": float(c0)},
        "constraints": {"a_clamp": [float(lo), float(hi)], "b_nonneg": True},
    }
    return RCModelParams(a=a, b=b, c=c, u_kind=str(u_kind), fit_n=int(calib_n)), yhat, r, meta


def model_quality(
    samples: List[ThermalSample],
    *,
    yhat: np.ndarray,
    residual: np.ndarray,
    fit_n: int,
) -> Dict[str, Any]:
    y = np.asarray([float(s.temp_c) for s in samples], dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    n = int(min(y.size, yhat.size, r.size))
    y = y[:n]
    yhat = yhat[:n]
    r = r[:n]
    fit_n = int(min(max(2, int(fit_n)), n))

    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        return float(np.mean((a - b) ** 2))

    calib_mse = _mse(y[1:fit_n], yhat[1:fit_n])
    eval_mse = _mse(y[fit_n:], yhat[fit_n:]) if fit_n < n else 0.0
    resid_mean = float(np.mean(r)) if r.size else 0.0
    resid_std = float(np.std(r, ddof=1) if r.size >= 2 else 0.0)
    resid_eval_std = float(np.std(r[fit_n:], ddof=1) if (n - fit_n) >= 2 else 0.0)

    # Residual autocorr on eval window (if any).
    def _autocorr(x: np.ndarray, lag: int) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        lag = int(lag)
        if x.size <= lag or lag <= 0:
            return 0.0
        a = x[:-lag]
        b = x[lag:]
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(a * b) / denom)

    eval_r = r[fit_n:] if fit_n < n else r
    ac = {str(l): _autocorr(eval_r, int(l)) for l in (1, 2, 5, 10)}

    adequate = True
    reasons = []
    if not (np.isfinite(calib_mse) and np.isfinite(eval_mse) and np.isfinite(resid_std)):
        adequate = False
        reasons.append("non_finite_metrics")
    if not (resid_std > 1e-6):
        adequate = False
        reasons.append("near_zero_residual_std")
    if n < 8:
        adequate = False
        reasons.append("too_few_samples")

    return {
        "n_samples": int(n),
        "fit_n": int(fit_n),
        "calib_mse": float(calib_mse),
        "eval_mse": float(eval_mse),
        "residual_mean": float(resid_mean),
        "residual_std": float(resid_std),
        "residual_eval_std": float(resid_eval_std),
        "residual_autocorr_eval": ac,
        "adequate": bool(adequate),
        "inadequacy_reasons": reasons,
    }


def write_rc_model_params(path: Path, params: RCModelParams, *, extra: Optional[Dict[str, Any]] = None) -> None:
    obj: Dict[str, Any] = {"rc_model": asdict(params)}
    if extra:
        obj.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
