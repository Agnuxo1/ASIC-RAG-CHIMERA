from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EntropySourceSpec:
    id: str
    available: bool
    trust_class: str  # PRNG | OS_ENTROPY | TIMING | CPU_CAPABILITY_ONLY | NETWORK_BEACON
    speed_estimate_bytes_per_sec: Optional[float]
    blocking_risk: str  # none|low|high
    notes: List[str]


def _has_os_getrandom() -> bool:
    return hasattr(os, "getrandom")


def _linux_cpu_flags() -> List[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for line in text.splitlines():
        if line.lower().startswith("flags"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return [p.strip() for p in parts[1].strip().split() if p.strip()]
    return []


def _estimate_bytes_per_sec_read(read_fn, *, n: int = 256 * 1024, max_s: float = 0.5) -> Optional[float]:
    n = int(n)
    if n <= 0:
        return None
    t0 = time.perf_counter()
    got = 0
    while got < n:
        chunk = min(32 * 1024, n - got)
        b = read_fn(int(chunk))
        got += len(b)
        if (time.perf_counter() - t0) > float(max_s):
            break
    dt = time.perf_counter() - t0
    if dt <= 0:
        return None
    return float(got / dt)


def discover_entropy_sources(*, beacon_url: Optional[str] = None) -> List[EntropySourceSpec]:
    specs: List[EntropySourceSpec] = []

    # Deterministic baseline always available.
    specs.append(
        EntropySourceSpec(
            id="prng",
            available=True,
            trust_class="PRNG",
            speed_estimate_bytes_per_sec=None,
            blocking_risk="none",
            notes=["Deterministic baseline (seeded)."],
        )
    )

    # os.urandom always present in Python.
    specs.append(
        EntropySourceSpec(
            id="os_urandom",
            available=True,
            trust_class="OS_ENTROPY",
            speed_estimate_bytes_per_sec=_estimate_bytes_per_sec_read(os.urandom),
            blocking_risk="low",
            notes=["Python os.urandom()."],
        )
    )

    # os.getrandom (Linux/Unix); flags availability differs by platform.
    if _has_os_getrandom():
        notes: List[str] = ["Python os.getrandom()."]
        flags = []
        for name in ("GRND_NONBLOCK", "GRND_RANDOM"):
            if hasattr(os, name):
                flags.append(name)
        if flags:
            notes.append(f"flags_available={','.join(flags)}")
        specs.append(
            EntropySourceSpec(
                id="os_getrandom",
                available=True,
                trust_class="OS_ENTROPY",
                speed_estimate_bytes_per_sec=_estimate_bytes_per_sec_read(os.getrandom),  # type: ignore[arg-type]
                blocking_risk="low",
                notes=notes,
            )
        )
    else:
        specs.append(
            EntropySourceSpec(
                id="os_getrandom",
                available=False,
                trust_class="OS_ENTROPY",
                speed_estimate_bytes_per_sec=None,
                blocking_risk="low",
                notes=["os.getrandom not available on this Python/platform."],
            )
        )

    # /dev/random: report presence, but default policy is to avoid using it (may block).
    dev_random = Path("/dev/random")
    specs.append(
        EntropySourceSpec(
            id="dev_random",
            available=dev_random.exists(),
            trust_class="OS_ENTROPY",
            speed_estimate_bytes_per_sec=None,
            blocking_risk="high",
            notes=["Linux/Unix special device; may block. Not used by default."],
        )
    )

    # Timing jitter: always "available" but can be slow/low-quality depending on clocksource.
    specs.append(
        EntropySourceSpec(
            id="timing_jitter",
            available=True,
            trust_class="TIMING",
            speed_estimate_bytes_per_sec=None,
            blocking_risk="none",
            notes=["Derived from perf_counter_ns jitter; machine-dependent and slow."],
        )
    )

    # CPU instruction capabilities (availability only; do not treat as a used entropy stream).
    flags = set(_linux_cpu_flags())
    specs.append(
        EntropySourceSpec(
            id="rdrand_capability",
            available=("rdrand" in flags),
            trust_class="CPU_CAPABILITY_ONLY",
            speed_estimate_bytes_per_sec=None,
            blocking_risk="none",
            notes=["Capability flag only; not used as an entropy source in this project."],
        )
    )
    specs.append(
        EntropySourceSpec(
            id="rdseed_capability",
            available=("rdseed" in flags),
            trust_class="CPU_CAPABILITY_ONLY",
            speed_estimate_bytes_per_sec=None,
            blocking_risk="none",
            notes=["Capability flag only; not used as an entropy source in this project."],
        )
    )

    # Optional network beacon.
    if beacon_url:
        specs.append(
            EntropySourceSpec(
                id="beacon",
                available=True,
                trust_class="NETWORK_BEACON",
                speed_estimate_bytes_per_sec=None,
                blocking_risk="low",
                notes=[f"url={beacon_url}"],
            )
        )
    else:
        specs.append(
            EntropySourceSpec(
                id="beacon",
                available=False,
                trust_class="NETWORK_BEACON",
                speed_estimate_bytes_per_sec=None,
                blocking_risk="low",
                notes=["disabled (no beacon_url provided)"],
            )
        )

    # Thermal residual (CPU telemetry) — best-effort, Linux-first.
    try:
        from chimera_handoff.entropy.thermal_telemetry import collect_thermal_telemetry, list_temperature_sensors

        sensors = list_temperature_sensors()
        if sensors:
            # Short viability check (per spec: can sample for a short window without exceptions).
            _samples, meta = collect_thermal_telemetry(duration_s=2.0, dt_s=0.1, temp_sensor="auto", use_rapl=True)
            specs.append(
                EntropySourceSpec(
                    id="thermal_residual",
                    available=True,
                    trust_class="CPU_TELEMETRY",
                    speed_estimate_bytes_per_sec=None,
                    blocking_risk="none",
                    notes=[
                        "CPU thermal telemetry available; residual stream requires fitting a simple thermal model.",
                        f"temp_path={meta.get('temp_path')}",
                        f"temp_label={dict(meta.get('temp_sensor', {})).get('label','')}",
                        f"rapl_energy_path={meta.get('rapl_energy_path')}",
                    ],
                )
            )
        else:
            specs.append(
                EntropySourceSpec(
                    id="thermal_residual",
                    available=False,
                    trust_class="CPU_TELEMETRY",
                    speed_estimate_bytes_per_sec=None,
                    blocking_risk="none",
                    notes=["no usable temperature sensors found under /sys"],
                )
            )
    except Exception as e:
        specs.append(
            EntropySourceSpec(
                id="thermal_residual",
                available=False,
                trust_class="CPU_TELEMETRY",
                speed_estimate_bytes_per_sec=None,
                blocking_risk="none",
                notes=[f"thermal discovery failed: {type(e).__name__}: {e}"],
            )
        )

    return specs


def platform_summary() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
    }
