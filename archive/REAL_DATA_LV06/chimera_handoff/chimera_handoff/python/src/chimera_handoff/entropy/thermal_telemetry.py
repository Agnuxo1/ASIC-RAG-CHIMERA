from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ThermalSample:
    t_ns: int
    temp_c: float
    util: float
    freq_mhz: float
    power_w: Optional[float]
    fan_rpm: Optional[float]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_int(path: Path) -> int:
    return int(_read_text(path).strip())


def _millideg_to_c(x: float) -> float:
    # Heuristic: sysfs temps are typically in millidegrees.
    if abs(x) > 200.0:
        return float(x) / 1000.0
    return float(x)


def _candidate_temp_paths() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # /sys/class/thermal/thermal_zone*/temp (+ type)
    base = Path("/sys/class/thermal")
    if base.exists():
        for tz in sorted(base.glob("thermal_zone*")):
            temp = tz / "temp"
            if not temp.exists():
                continue
            typ = tz / "type"
            label = _read_text(typ) if typ.exists() else ""
            out.append({"path": str(temp), "kind": "thermal_zone", "label": label})

    # /sys/class/hwmon/hwmon*/temp*_input (+ temp*_label)
    hw = Path("/sys/class/hwmon")
    if hw.exists():
        for hwmon in sorted(hw.glob("hwmon*")):
            for inp in sorted(hwmon.glob("temp*_input")):
                label_p = inp.with_name(inp.name.replace("_input", "_label"))
                label = _read_text(label_p) if label_p.exists() else ""
                out.append({"path": str(inp), "kind": "hwmon", "label": label, "hwmon": str(hwmon)})

    return out


def list_temperature_sensors() -> List[Dict[str, Any]]:
    sensors = []
    for c in _candidate_temp_paths():
        p = Path(str(c["path"]))
        try:
            v = _millideg_to_c(float(_read_int(p)))
        except Exception:
            continue
        if not (np.isfinite(v) and -10.0 <= float(v) <= 130.0):
            continue
        sensors.append({**c, "temp_c_now": float(v)})
    return sensors


def choose_temperature_sensor(temp_sensor: str = "auto") -> Dict[str, Any]:
    sensors = list_temperature_sensors()
    if not sensors:
        raise RuntimeError("no usable temperature sensors found under /sys")

    s = str(temp_sensor).strip()
    if s and s.lower() != "auto":
        p = Path(s)
        for c in sensors:
            if Path(str(c["path"])) == p:
                return c
        raise RuntimeError(f"requested temp_sensor path not usable: {s}")

    def score(x: Dict[str, Any]) -> Tuple[int, float]:
        label = str(x.get("label", "")).lower()
        good = any(k in label for k in ["package", "tctl", "cpu", "core"])
        return (1 if good else 0, float(x.get("temp_c_now", 0.0)))

    # Prefer (heuristically) package/cpu labels, then higher current temp.
    return sorted(sensors, key=score, reverse=True)[0]


def _read_cpu_stat() -> Tuple[int, int]:
    """
    Returns (total_jiffies, idle_jiffies) from /proc/stat.
    """
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return 0, 0
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return 0, 0
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    return int(total), int(idle)


def _cpu_util_over_dt(prev: Tuple[int, int], cur: Tuple[int, int]) -> float:
    t0, i0 = prev
    t1, i1 = cur
    dt = max(1, int(t1 - t0))
    didle = max(0, int(i1 - i0))
    util = 1.0 - float(didle) / float(dt)
    return float(np.clip(util, 0.0, 1.0))


def _read_cpu_freq_mhz() -> float:
    # Best-effort: cpufreq scaling_cur_freq (kHz) or /proc/cpuinfo fallback.
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if p.exists():
        try:
            khz = float(_read_int(p))
            return float(khz / 1000.0)
        except Exception:
            pass
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "cpu mhz" in line.lower():
                _, rhs = line.split(":", 1)
                return float(rhs.strip())
    except Exception:
        pass
    return 0.0


def _list_fan_rpm_paths() -> List[Path]:
    out: List[Path] = []
    hw = Path("/sys/class/hwmon")
    if not hw.exists():
        return out
    for hwmon in sorted(hw.glob("hwmon*")):
        out.extend(sorted(hwmon.glob("fan*_input")))
    return out


def _read_first_fan_rpm() -> Optional[float]:
    for p in _list_fan_rpm_paths():
        try:
            v = float(_read_int(p))
        except Exception:
            continue
        if np.isfinite(v) and v >= 0:
            return float(v)
    return None


def _list_rapl_energy_paths() -> List[Path]:
    base = Path("/sys/class/powercap")
    out: List[Path] = []
    if not base.exists():
        return out
    for p in sorted(base.glob("intel-rapl:*")):
        e = p / "energy_uj"
        if e.exists():
            out.append(e)
    return out


def collect_thermal_telemetry(
    *,
    duration_s: float,
    dt_s: float,
    temp_sensor: str = "auto",
    use_rapl: bool = True,
) -> Tuple[List[ThermalSample], Dict[str, Any]]:
    dt_s = float(dt_s)
    duration_s = float(duration_s)
    if not (dt_s > 0):
        raise ValueError("dt_s must be positive")
    if not (duration_s > 0):
        raise ValueError("duration_s must be positive")

    sensor = choose_temperature_sensor(temp_sensor=str(temp_sensor))
    temp_path = Path(str(sensor["path"]))

    rapl_paths = _list_rapl_energy_paths() if bool(use_rapl) else []
    rapl_path = rapl_paths[0] if rapl_paths else None

    n = int(max(2, round(duration_s / dt_s)))
    samples: List[ThermalSample] = []

    cpu_prev = _read_cpu_stat()
    energy_prev = None
    t0_ns = time.monotonic_ns()
    next_ns = int(t0_ns)
    for i in range(int(n)):
        # Schedule.
        now_ns = time.monotonic_ns()
        if now_ns < next_ns:
            time.sleep((next_ns - now_ns) / 1e9)
        t_ns = time.monotonic_ns()

        # Read sensors.
        temp_c = _millideg_to_c(float(_read_int(temp_path)))
        cpu_cur = _read_cpu_stat()
        util = _cpu_util_over_dt(cpu_prev, cpu_cur) if i > 0 else 0.0
        cpu_prev = cpu_cur

        freq_mhz = float(_read_cpu_freq_mhz())

        power_w: Optional[float] = None
        if rapl_path is not None and rapl_path.exists():
            try:
                e = int(_read_int(rapl_path))
                if energy_prev is not None:
                    de_uj = int(e - energy_prev)
                    if de_uj >= 0:
                        power_w = float((de_uj / 1e6) / dt_s)
                energy_prev = int(e)
            except Exception:
                power_w = None

        fan_rpm = _read_first_fan_rpm()

        samples.append(
            ThermalSample(
                t_ns=int(t_ns),
                temp_c=float(temp_c),
                util=float(util),
                freq_mhz=float(freq_mhz),
                power_w=float(power_w) if power_w is not None else None,
                fan_rpm=float(fan_rpm) if fan_rpm is not None else None,
            )
        )

        next_ns = int(t0_ns + int(round((i + 1) * dt_s * 1e9)))

    # dt stats (observed, from timestamps).
    t_arr = np.asarray([int(s.t_ns) for s in samples], dtype=np.int64)
    dts = np.diff(t_arr).astype(np.float64) / 1e9
    if dts.size:
        dt_stats = {
            "dt_s_nominal": float(dt_s),
            "dt_s_mean": float(np.mean(dts)),
            "dt_s_p50": float(np.quantile(dts, 0.50)),
            "dt_s_p95": float(np.quantile(dts, 0.95)),
            "dt_s_max": float(np.max(dts)),
            "dt_s_min": float(np.min(dts)),
            "n_intervals": int(dts.size),
            "n_nonpositive": int(np.sum(dts <= 0.0)),
        }
    else:
        dt_stats = {
            "dt_s_nominal": float(dt_s),
            "dt_s_mean": 0.0,
            "dt_s_p50": 0.0,
            "dt_s_p95": 0.0,
            "dt_s_max": 0.0,
            "dt_s_min": 0.0,
            "n_intervals": 0,
            "n_nonpositive": 0,
        }

    meta: Dict[str, Any] = {
        "duration_s": float(duration_s),
        "dt_s": float(dt_s),
        "n_samples": int(len(samples)),
        "dt_stats": dict(dt_stats),
        "temp_sensor": dict(sensor),
        "temp_path": str(temp_path),
        "rapl_energy_path": str(rapl_path) if rapl_path is not None else None,
        "fan_paths_found": [str(p) for p in _list_fan_rpm_paths()],
    }
    return samples, meta


def write_thermal_telemetry_csv(path: Path, samples: List[ThermalSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["t_ns", "temp_c", "util", "freq_mhz", "power_w", "fan_rpm"],
        )
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))
