from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.thermal_config import ThermalConfig
from chimera_handoff.entropy.thermal_model import fit_first_order_rc, model_quality, write_rc_model_params
from chimera_handoff.entropy.thermal_telemetry import ThermalSample, collect_thermal_telemetry, write_thermal_telemetry_csv


class ThermalResidualSource:
    id: str = "thermal_residual"

    def __init__(
        self,
        *,
        seed: int = 0,
        config: Optional[ThermalConfig] = None,
        out_dir: Optional[Path] = None,
    ) -> None:
        self._seed = int(seed)
        self._cfg = (config or ThermalConfig()).sanitized()
        self._out_dir = Path(out_dir) if out_dir is not None else None
        self._ready = False

        self._samples: List[ThermalSample] = []
        self._telemetry_meta: Dict[str, Any] = {}
        self._model_quality: Dict[str, Any] = {}
        self._resid_raw: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._resid_norm: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._resid_stats: Dict[str, float] = {}
        self._cursor = 0
        self._rng = np.random.default_rng(int(self._seed) or 1)

    def stream_info(self) -> Dict[str, Any]:
        return {
            "source_id": self.id,
            "seed": int(self._seed),
            "thermal": {
                "dt_s": float(self._cfg.dt_s),
                "duration_s": float(self._cfg.duration_s),
                "calib_s": float(self._cfg.calib_s),
                "temp_sensor": str(self._cfg.temp_sensor),
                "use_rapl": bool(self._cfg.use_rapl),
                "clip_k": float(self._cfg.clip_k),
                "intervention": str(self._cfg.intervention),
                "intervention_threads": int(self._cfg.intervention_threads),
                "intervention_duty": float(self._cfg.intervention_duty),
                "intervention_period_s": float(self._cfg.intervention_period_s),
                "intervention_warmup_s": float(self._cfg.intervention_warmup_s),
            },
            "model_quality": dict(self._model_quality) if self._model_quality else None,
            "residual_stats": dict(self._resid_stats) if self._resid_stats else None,
        }

    @classmethod
    def from_samples(
        cls,
        samples: List[ThermalSample],
        *,
        seed: int = 0,
        config: Optional[ThermalConfig] = None,
        out_dir: Optional[Path] = None,
    ) -> ThermalResidualSource:
        obj = cls(seed=seed, config=config, out_dir=out_dir)
        obj._samples = list(samples)
        obj._telemetry_meta = {"injected_samples": True, "n_samples": int(len(samples))}
        obj._build_from_samples()
        obj._ready = True
        return obj

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        intervention_meta = {"mode": "none"}
        if str(self._cfg.intervention).lower().strip() not in {"none", ""}:
            from chimera_handoff.experiments.interventions import InterventionController

            ctrl = InterventionController(
                mode=str(self._cfg.intervention),
                threads=int(self._cfg.intervention_threads),
                duty=float(self._cfg.intervention_duty),
                period_s=float(self._cfg.intervention_period_s),
                warmup_s=float(self._cfg.intervention_warmup_s),
            )
            intervention_meta = ctrl.start()
            try:
                samples, meta = collect_thermal_telemetry(
                    duration_s=float(self._cfg.duration_s),
                    dt_s=float(self._cfg.dt_s),
                    temp_sensor=str(self._cfg.temp_sensor),
                    use_rapl=bool(self._cfg.use_rapl),
                )
            finally:
                intervention_meta.update(ctrl.stop())
        else:
            samples, meta = collect_thermal_telemetry(
                duration_s=float(self._cfg.duration_s),
                dt_s=float(self._cfg.dt_s),
                temp_sensor=str(self._cfg.temp_sensor),
                use_rapl=bool(self._cfg.use_rapl),
            )
        self._samples = list(samples)
        self._telemetry_meta = {**dict(meta), "intervention": dict(intervention_meta)}
        self._build_from_samples()
        self._ready = True

    def _build_from_samples(self) -> None:
        if len(self._samples) < 4:
            raise RuntimeError("thermal telemetry stream too short to fit model")

        calib_n = int(max(3, round(float(self._cfg.calib_s) / float(self._cfg.dt_s))))
        params, yhat, r, fit_meta = fit_first_order_rc(self._samples, calib_n=int(calib_n))
        self._model_quality = model_quality(self._samples, yhat=yhat, residual=r, fit_n=int(params.fit_n))

        # Standardize using calibration residual stats.
        r_cal = np.asarray(r[: params.fit_n], dtype=np.float64).reshape(-1)
        mu = float(np.mean(r_cal)) if r_cal.size else 0.0
        sd = float(np.std(r_cal, ddof=1) if r_cal.size >= 2 else 0.0)
        sd = float(sd if sd > 1e-9 else 1.0)

        r_norm = ((np.asarray(r, dtype=np.float64) - mu) / sd).astype(np.float32)
        r_norm = np.clip(r_norm, -float(self._cfg.clip_k), float(self._cfg.clip_k)).astype(np.float32)

        self._resid_raw = np.asarray(r, dtype=np.float32)
        self._resid_norm = np.asarray(r_norm, dtype=np.float32)
        self._resid_stats = {
            "calib_mean": float(mu),
            "calib_std": float(sd),
            "clip_k": float(self._cfg.clip_k),
            "resid_norm_mean": float(np.mean(self._resid_norm)) if self._resid_norm.size else 0.0,
            "resid_norm_std": float(np.std(self._resid_norm, ddof=1)) if self._resid_norm.size >= 2 else 0.0,
        }
        self._cursor = 0

        # Seed RNG deterministically from observed residuals + configured seed.
        h = hashlib.sha256()
        h.update(self._resid_norm.tobytes())
        h.update(int(self._seed).to_bytes(8, "little", signed=True))
        seed64 = int.from_bytes(h.digest()[:8], "little", signed=False) % (2**63 - 1)
        self._rng = np.random.default_rng(int(seed64) or 1)

        # Write artifacts (best-effort).
        if self._out_dir is not None:
            out = self._out_dir / "thermal"
            out.mkdir(parents=True, exist_ok=True)
            write_thermal_telemetry_csv(out / "thermal_telemetry.csv", self._samples)
            (out / "thermal_telemetry_meta.json").write_text(
                json.dumps(self._telemetry_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_rc_model_params(out / "thermal_model_params.json", params, extra={"fit_meta": fit_meta})
            self._write_residual_csv(out / "thermal_residual.csv", yhat=yhat, r=np.asarray(r, dtype=np.float64))
            (out / "thermal_residual_meta.json").write_text(
                json.dumps({"residual_stats": dict(self._resid_stats), "model_quality": dict(self._model_quality)}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out / "thermal_model_quality.json").write_text(
                json.dumps(dict(self._model_quality), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _write_residual_csv(self, path: Path, *, yhat: np.ndarray, r: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        y = np.asarray([float(s.temp_c) for s in self._samples], dtype=np.float64)
        u = np.asarray([0.0 if s.power_w is None else float(s.power_w) for s in self._samples], dtype=np.float64)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_ns", "temp_c", "temp_hat_c", "residual_c", "power_w"])
            w.writeheader()
            for i, s in enumerate(self._samples):
                w.writerow(
                    {
                        "t_ns": int(s.t_ns),
                        "temp_c": float(y[i]),
                        "temp_hat_c": float(yhat[i]) if i < len(yhat) else 0.0,
                        "residual_c": float(r[i]) if i < len(r) else 0.0,
                        "power_w": float(u[i]),
                    }
                )

    def residual_norm_series(self) -> np.ndarray:
        self._ensure_ready()
        return self._resid_norm.astype(np.float32, copy=True)

    def residual_raw_series(self) -> np.ndarray:
        self._ensure_ready()
        return self._resid_raw.astype(np.float32, copy=True)

    def timestamps_ns(self) -> np.ndarray:
        self._ensure_ready()
        return np.asarray([int(s.t_ns) for s in self._samples], dtype=np.int64)

    def read_floats(self, n: int) -> np.ndarray:
        self._ensure_ready()
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        buf = self._resid_norm.reshape(-1)
        if buf.size == 0:
            return np.zeros((n,), dtype=np.float32)
        out = np.empty((n,), dtype=np.float32)
        for i in range(int(n)):
            out[i] = float(buf[self._cursor % int(buf.size)])
            self._cursor = int(self._cursor + 1)
        return out

    def read_normal_f32(self, n: int) -> np.ndarray:
        # IID sampling (with replacement) for injection-style use-cases.
        self._ensure_ready()
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        buf = self._resid_norm.reshape(-1)
        if buf.size == 0:
            return np.zeros((n,), dtype=np.float32)
        idx = self._rng.integers(0, int(buf.size), size=int(n), endpoint=False)
        return buf[idx].astype(np.float32, copy=False)

    # Optional compatibility: some tooling expects read_bytes.
    def read_bytes(self, n: int) -> bytes:  # pragma: no cover
        raise NotImplementedError("thermal_residual is a float stream; use read_floats/read_normal_f32")
