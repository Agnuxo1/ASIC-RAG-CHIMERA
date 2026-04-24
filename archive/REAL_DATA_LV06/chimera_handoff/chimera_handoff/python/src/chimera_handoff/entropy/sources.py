from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


class EntropySource:
    id: str

    def read_bytes(self, n: int) -> bytes:  # pragma: no cover (interface)
        raise NotImplementedError

    def read_u64(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.uint64)
        b = self.read_bytes(8 * n)
        if len(b) < 8 * n:
            b = b + b"\x00" * (8 * n - len(b))
        return np.frombuffer(b[: 8 * n], dtype=np.uint64)

    def read_uniform_f32(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        # Use uint32 to map into [0,1).
        b = self.read_bytes(4 * n)
        if len(b) < 4 * n:
            b = b + b"\x00" * (4 * n - len(b))
        u = np.frombuffer(b[: 4 * n], dtype=np.uint32).astype(np.float64)
        x = (u / float(2**32)).astype(np.float32)
        return x

    def read_normal_f32(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        # Box-Muller using uniforms from bytes.
        m = (n + 1) // 2
        u1 = self.read_uniform_f32(m).astype(np.float64)
        u2 = self.read_uniform_f32(m).astype(np.float64)
        u1 = np.clip(u1, 1e-12, 1.0 - 1e-12)
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        z0 = r * np.cos(theta)
        z1 = r * np.sin(theta)
        out = np.empty((2 * m,), dtype=np.float64)
        out[0::2] = z0
        out[1::2] = z1
        return out[:n].astype(np.float32)


@dataclass
class PRNGSource(EntropySource):
    seed: int
    id: str = "prng"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    def read_bytes(self, n: int) -> bytes:
        n = int(n)
        if n <= 0:
            return b""
        arr = self._rng.integers(0, 256, size=n, dtype=np.uint8)
        return arr.tobytes()


class OSUrandomSource(EntropySource):
    id: str = "os_urandom"

    def read_bytes(self, n: int) -> bytes:
        n = int(n)
        if n <= 0:
            return b""
        return os.urandom(n)


@dataclass
class OSGetrandomSource(EntropySource):
    flags: int = 0
    id: str = "os_getrandom"

    def read_bytes(self, n: int) -> bytes:
        if not hasattr(os, "getrandom"):
            raise RuntimeError("os.getrandom not available on this platform")
        n = int(n)
        if n <= 0:
            return b""
        return os.getrandom(n, self.flags)  # type: ignore[attr-defined]


class DevRandomSource(EntropySource):
    id: str = "dev_random"

    def __init__(self) -> None:
        self._path = Path("/dev/random")
        if not self._path.exists():
            raise FileNotFoundError(str(self._path))

    def read_bytes(self, n: int) -> bytes:
        n = int(n)
        if n <= 0:
            return b""
        with self._path.open("rb", buffering=0) as f:
            return f.read(n)


@dataclass
class TimingJitterSource(EntropySource):
    id: str = "timing_jitter"
    spin_iters: int = 256

    def _one_u64(self) -> int:
        t0 = time.perf_counter_ns()
        x = 0
        for i in range(int(self.spin_iters)):
            t1 = time.perf_counter_ns()
            x = (x ^ (t1 - t0) ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            t0 = t1
        return int(x)

    def read_bytes(self, n: int) -> bytes:
        n = int(n)
        if n <= 0:
            return b""
        out = bytearray(n)
        i = 0
        while i < n:
            x = self._one_u64()
            b = int(x).to_bytes(8, byteorder="little", signed=False)
            take = min(8, n - i)
            out[i : i + take] = b[:take]
            i += take
        return bytes(out)


@dataclass
class BeaconSource(EntropySource):
    url: str
    timeout_s: float = 2.0
    id: str = "beacon"

    def read_bytes(self, n: int) -> bytes:
        n = int(n)
        if n <= 0:
            return b""
        # Best-effort: fetch and hash to bytes.
        req = urllib.request.Request(self.url, headers={"User-Agent": "heyting-reservoir/1.0"})
        with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
            data = resp.read()
        # Expand deterministically to requested length by rehashing.
        import hashlib

        out = bytearray()
        counter = 0
        while len(out) < n:
            h = hashlib.sha256()
            h.update(data)
            h.update(counter.to_bytes(8, "little", signed=False))
            out.extend(h.digest())
            counter += 1
        return bytes(out[:n])


def make_source(
    source_id: str,
    *,
    seed: int = 0,
    getrandom_flags: int = 0,
    beacon_url: Optional[str] = None,
    # Thermal residual config (used only when source_id == thermal_residual).
    thermal_dt: float = 0.05,
    thermal_duration: float = 20.0,
    thermal_calib_seconds: float = 5.0,
    thermal_temp_sensor: str = "auto",
    thermal_use_rapl: bool = True,
    thermal_clip_k: float = 5.0,
    thermal_intervention: str = "none",
    intervention_threads: int = 1,
    intervention_duty: float = 0.5,
    intervention_period: float = 1.0,
    intervention_warmup: float = 0.0,
    thermal_out_dir: Optional[Union[str, Path]] = None,
) -> object:
    s = str(source_id).lower().strip()
    if s == "prng":
        return PRNGSource(seed=int(seed))
    if s in {"os_urandom", "urandom"}:
        return OSUrandomSource()
    if s == "os_getrandom":
        return OSGetrandomSource(flags=int(getrandom_flags))
    if s == "dev_random":
        return DevRandomSource()
    if s in {"timing_jitter", "timing"}:
        return TimingJitterSource()
    if s == "beacon":
        if not beacon_url:
            raise ValueError("beacon_url is required for beacon source")
        return BeaconSource(url=str(beacon_url))
    if s in {"thermal_residual", "thermal"}:
        from chimera_handoff.entropy.thermal_config import ThermalConfig
        from chimera_handoff.entropy.thermal_residual_source import ThermalResidualSource

        cfg = ThermalConfig(
            dt_s=float(thermal_dt),
            duration_s=float(thermal_duration),
            calib_s=float(thermal_calib_seconds),
            temp_sensor=str(thermal_temp_sensor),
            use_rapl=bool(thermal_use_rapl),
            clip_k=float(thermal_clip_k),
            intervention=str(thermal_intervention),
            intervention_threads=int(intervention_threads),
            intervention_duty=float(intervention_duty),
            intervention_period_s=float(intervention_period),
            intervention_warmup_s=float(intervention_warmup),
        ).sanitized()
        out_dir = Path(str(thermal_out_dir)) if thermal_out_dir is not None else None
        return ThermalResidualSource(seed=int(seed), config=cfg, out_dir=out_dir)
    raise ValueError(f"unknown entropy source id={source_id!r}")
