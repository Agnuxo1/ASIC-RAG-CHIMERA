from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThermalConfig:
    dt_s: float = 0.05
    duration_s: float = 20.0
    calib_s: float = 5.0
    temp_sensor: str = "auto"  # "auto" or explicit sysfs path
    use_rapl: bool = True
    clip_k: float = 5.0

    # Controlled intervention (best-effort): none|cpu_burn|cpu_duty
    intervention: str = "none"
    intervention_threads: int = 1
    intervention_duty: float = 0.5
    intervention_period_s: float = 1.0
    intervention_warmup_s: float = 0.0

    # Hard safety caps.
    max_duration_s: float = 60.0
    min_duration_s: float = 2.0
    min_dt_s: float = 0.01
    max_dt_s: float = 1.0

    def sanitized(self) -> ThermalConfig:
        dt = min(max(float(self.dt_s), float(self.min_dt_s)), float(self.max_dt_s))
        dur = min(max(float(self.duration_s), float(self.min_duration_s)), float(self.max_duration_s))
        calib = min(max(float(self.calib_s), dt), max(dt, dur - dt))
        clip_k = float(self.clip_k)
        if not (clip_k > 0.0):
            clip_k = 5.0
        m = str(self.intervention).lower().strip()
        if m not in {"none", "cpu_burn", "cpu_duty"}:
            m = "none"
        threads = int(max(1, int(self.intervention_threads)))
        duty = float(self.intervention_duty)
        if duty < 0.0:
            duty = 0.0
        if duty > 1.0:
            duty = 1.0
        period_s = float(max(0.05, float(self.intervention_period_s)))
        warmup_s = float(max(0.0, float(self.intervention_warmup_s)))
        return ThermalConfig(
            dt_s=float(dt),
            duration_s=float(dur),
            calib_s=float(calib),
            temp_sensor=str(self.temp_sensor),
            use_rapl=bool(self.use_rapl),
            clip_k=float(clip_k),
            intervention=str(m),
            intervention_threads=int(threads),
            intervention_duty=float(duty),
            intervention_period_s=float(period_s),
            intervention_warmup_s=float(warmup_s),
            max_duration_s=float(self.max_duration_s),
            min_duration_s=float(self.min_duration_s),
            min_dt_s=float(self.min_dt_s),
            max_dt_s=float(self.max_dt_s),
        )


def parse_bool(x: object) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def maybe_path_str(x: Optional[object]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None
