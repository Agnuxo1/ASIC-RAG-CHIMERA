from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from chimera_handoff.system.discovery import EntropySourceSpec, discover_entropy_sources, platform_summary


def _perf_counter_resolution_ns(*, iters: int = 10_000) -> Dict[str, Any]:
    prev = time.perf_counter_ns()
    best = None
    deltas: List[int] = []
    for _ in range(int(iters)):
        cur = time.perf_counter_ns()
        d = int(cur - prev)
        if d > 0:
            deltas.append(d)
            best = d if best is None else min(best, d)
        prev = cur
    return {"min_positive_delta_ns": int(best or 0), "samples": int(len(deltas))}


def _cpu_info_linux() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return info
    model = None
    flags = None
    for line in txt.splitlines():
        if line.lower().startswith("model name"):
            model = line.split(":", 1)[1].strip() if ":" in line else None
        if line.lower().startswith("flags"):
            flags = line.split(":", 1)[1].strip().split() if ":" in line else None
        if model and flags:
            break
    if model:
        info["model_name"] = model
    if flags:
        info["flags"] = flags
    return info


def write_system_profile(out_path: Path, *, beacon_url: Optional[str] = None) -> Dict[str, Any]:
    sources = discover_entropy_sources(beacon_url=beacon_url)
    uname_obj = None
    if hasattr(os, "uname"):
        u = os.uname()
        uname_obj = {
            "sysname": getattr(u, "sysname", None),
            "nodename": getattr(u, "nodename", None),
            "release": getattr(u, "release", None),
            "version": getattr(u, "version", None),
            "machine": getattr(u, "machine", None),
        }
    obj: Dict[str, Any] = {
        "platform": platform_summary(),
        "uname": uname_obj,
        "cpu": _cpu_info_linux() if platform.system().lower() == "linux" else {},
        "cores": {"os_cpu_count": int(os.cpu_count() or 0)},
        "timers": {"perf_counter_ns": _perf_counter_resolution_ns()},
        "entropy_sources": [s.__dict__ for s in sources],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return obj
