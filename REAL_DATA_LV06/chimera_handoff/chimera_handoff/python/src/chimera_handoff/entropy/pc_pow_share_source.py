from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimera_handoff.entropy.chronos_event_types import EventRecord
from chimera_handoff.util.paths import ensure_out_dir


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _meets_leading_zero_bits(digest: bytes, *, difficulty_bits: int) -> bool:
    b = int(max(0, difficulty_bits))
    if b <= 0:
        return True
    full = b // 8
    rem = b % 8
    if full:
        if any(x != 0 for x in digest[:full]):
            return False
    if rem == 0:
        return True
    nxt = digest[full]
    mask = 0xFF & (0xFF << (8 - rem))
    return (nxt & mask) == 0


def _apply_confound_jitter(t_ns: np.ndarray, *, jitter_ms: float, seed: int) -> np.ndarray:
    t = np.asarray(t_ns, dtype=np.int64).copy()
    j = float(jitter_ms)
    if not (j > 0):
        return t
    rng = np.random.default_rng(int(seed) + 991)
    amp = int(round(j * 1e6))
    noise = rng.integers(-amp, amp + 1, size=int(t.size), dtype=np.int64)
    t = t + noise
    # Enforce strictly increasing.
    for i in range(1, int(t.size)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
    return t


def _apply_confound_batching(t_ns: np.ndarray, *, flush_ms: float) -> np.ndarray:
    t = np.asarray(t_ns, dtype=np.int64).copy()
    if t.size == 0:
        return t
    f = float(flush_ms)
    if not (f > 0):
        return t
    flush_ns = int(round(f * 1e6))
    t0 = int(t[0])
    # Assign each event to a flush bucket; observed time is the flush time.
    bucket = ((t - t0) // max(1, flush_ns)).astype(np.int64)
    out = np.empty_like(t)
    cur = 0
    while cur < int(t.size):
        b = int(bucket[cur])
        j = cur
        while j < int(t.size) and int(bucket[j]) == b:
            j += 1
        flush_time = t0 + (b + 1) * flush_ns
        # Give events in-bucket a tiny increasing offset to preserve strict monotonicity.
        for k in range(cur, j):
            out[k] = int(flush_time + (k - cur))
        cur = j
    return out


@dataclass(frozen=True)
class PcPoWShareConfig:
    schema_version: str = "0.1"
    seed: int = 0
    duration_s: float = 120.0
    difficulty_bits: int = 20
    backend: str = "cpu"  # cpu|gpu (gpu optional; cpu only implemented)
    threads: int = 1

    header_mode: str = "seeded_prefix"  # reserved for ASIC parity
    header_prefix_bytes: int = 80

    intervention: str = "none"  # none|heartbeat
    heartbeat_hz: float = 2.4
    heartbeat_duty: float = 0.5

    confound: str = "none"  # none|batching|jitter
    batch_flush_ms: float = 200.0
    jitter_ms: float = 2.0

    def sanitized(self) -> PcPoWShareConfig:
        return PcPoWShareConfig(
            schema_version=str(self.schema_version),
            seed=int(self.seed),
            duration_s=float(max(0.0, float(self.duration_s))),
            difficulty_bits=int(max(0, int(self.difficulty_bits))),
            backend=str(self.backend).lower().strip() or "cpu",
            threads=int(max(1, int(self.threads))),
            header_mode=str(self.header_mode),
            header_prefix_bytes=int(max(1, int(self.header_prefix_bytes))),
            intervention=str(self.intervention).lower().strip() or "none",
            heartbeat_hz=float(self.heartbeat_hz),
            heartbeat_duty=float(self.heartbeat_duty),
            confound=str(self.confound).lower().strip() or "none",
            batch_flush_ms=float(self.batch_flush_ms),
            jitter_ms=float(self.jitter_ms),
        )


class PcPoWShareEventSource:
    """
    Network-free share-like event generator on a local PC:
    a "share event" occurs when SHA-256(header_prefix || nonce) has leading zero bits.
    """

    id: str = "pc_pow_share_events"

    def __init__(self, *, config: Optional[PcPoWShareConfig] = None) -> None:
        self.cfg = (config or PcPoWShareConfig()).sanitized()

    def _make_prefix(self) -> bytes:
        # Deterministic prefix from seed; matches "header-like" behavior.
        rng = np.random.default_rng(int(self.cfg.seed) + 17)
        x = rng.integers(0, 256, size=int(self.cfg.header_prefix_bytes), dtype=np.uint8)
        return bytes(x.tolist())

    def run(self, *, out_dir: Path) -> Tuple[List[EventRecord], Dict[str, Any]]:
        out = ensure_out_dir(Path(out_dir))

        if str(self.cfg.backend) not in {"cpu"}:
            raise NotImplementedError("only cpu backend is implemented in this repo")
        if int(self.cfg.threads) != 1:
            # Threads are deliberately not enabled: Python-level scheduling affects timing,
            # and we want a clean readiness baseline for ASIC-parity analysis.
            raise NotImplementedError("threads>1 not implemented (use --threads 1)")

        prefix = self._make_prefix()
        diff = int(self.cfg.difficulty_bits)

        intervention = str(self.cfg.intervention).lower().strip()
        confound = str(self.cfg.confound).lower().strip()
        hb_hz = float(self.cfg.heartbeat_hz)
        hb_duty = float(self.cfg.heartbeat_duty)

        t_start = time.monotonic_ns()
        t_stop = t_start + int(round(float(self.cfg.duration_s) * 1e9))

        protocol = {
            "schema_version": str(self.cfg.schema_version),
            "source_id": self.id,
            "seed": int(self.cfg.seed),
            "backend": "cpu",
            "threads": 1,
            "duration_s": float(self.cfg.duration_s),
            "difficulty_bits": int(diff),
            "header_mode": str(self.cfg.header_mode),
            "header_prefix_bytes": int(self.cfg.header_prefix_bytes),
            "intervention": {"mode": intervention, "heartbeat_hz": float(hb_hz), "heartbeat_duty": float(hb_duty)},
            "confound": {"mode": confound, "batch_flush_ms": float(self.cfg.batch_flush_ms), "jitter_ms": float(self.cfg.jitter_ms)},
        }
        _write_json(out / "protocol.json", protocol)

        events: List[EventRecord] = []
        attempt_rows: List[Dict[str, Any]] = []

        nonce = 0
        attempts_total = 0
        attempts_since_prev = 0

        # Attempt rate logging.
        next_rate_log = t_start + int(1e9)
        last_attempts_total = 0

        # Heartbeat gating schedule.
        if intervention == "heartbeat" and hb_hz > 0:
            period_ns = int(round(1e9 / float(hb_hz)))
            on_ns = int(round(float(hb_duty) * float(period_ns)))
            on_ns = int(max(0, min(period_ns, on_ns)))
        else:
            period_ns = 0
            on_ns = 0

        t_phase0 = t_start

        while True:
            t_now = time.monotonic_ns()
            if t_now >= t_stop:
                break

            if period_ns > 0:
                phase = int((t_now - t_phase0) // max(1, period_ns))
                t_phase_start = t_phase0 + phase * period_ns
                t_on_end = t_phase_start + on_ns
                t_phase_end = t_phase_start + period_ns
                if t_now >= t_on_end:
                    # Off window: sleep until next phase start.
                    sleep_ns = max(0, t_phase_end - t_now)
                    time.sleep(float(sleep_ns) * 1e-9)
                    continue

            digest = hashlib.sha256(prefix + int(nonce).to_bytes(8, "little", signed=False)).digest()
            nonce += 1
            attempts_total += 1
            attempts_since_prev += 1

            if _meets_leading_zero_bits(digest, difficulty_bits=int(diff)):
                t_ev = time.monotonic_ns()
                events.append(
                    EventRecord(
                        t_ns=int(t_ev),
                        nonce=int(nonce - 1),
                        hash_hex=digest.hex(),
                        difficulty_bits=int(diff),
                        attempts_since_prev=int(attempts_since_prev),
                        backend="cpu",
                        notes={"t_true_ns": int(t_ev)},
                    )
                )
                attempts_since_prev = 0

            # Log attempt rate once per second.
            if t_now >= next_rate_log:
                d_attempts = int(attempts_total - last_attempts_total)
                attempt_rows.append(
                    {
                        "t_ns": int(t_now),
                        "attempts_total": int(attempts_total),
                        "attempts_per_s": int(d_attempts),
                    }
                )
                last_attempts_total = int(attempts_total)
                next_rate_log += int(1e9)

        # Apply confounds to observed timestamps.
        t_true = np.asarray([int(e.t_ns) for e in events], dtype=np.int64)
        if confound == "jitter":
            t_obs = _apply_confound_jitter(t_true, jitter_ms=float(self.cfg.jitter_ms), seed=int(self.cfg.seed))
        elif confound == "batching":
            t_obs = _apply_confound_batching(t_true, flush_ms=float(self.cfg.batch_flush_ms))
        else:
            t_obs = t_true

        # Update EventRecord list with observed timestamps.
        updated: List[EventRecord] = []
        for e, to in zip(events, t_obs.tolist()):
            updated.append(
                EventRecord(
                    t_ns=int(to),
                    nonce=int(e.nonce),
                    hash_hex=str(e.hash_hex),
                    difficulty_bits=int(e.difficulty_bits),
                    attempts_since_prev=int(e.attempts_since_prev),
                    backend=str(e.backend),
                    notes=dict(e.notes),
                )
            )
        events = updated

        # Write CSV artifacts.
        self._write_events_csv(out / "events.csv", events)
        self._write_attempt_rate_csv(out / "attempt_rate.csv", attempt_rows)
        _write_json(out / "counts.json", {"schema_version": str(self.cfg.schema_version), "n_events": int(len(events)), "n_attempts": int(attempts_total)})
        return events, protocol

    def _write_events_csv(self, path: Path, events: List[EventRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "t_ns",
                    "nonce",
                    "hash_hex",
                    "difficulty_bits",
                    "attempts_since_prev",
                    "backend",
                    "notes_json",
                ],
            )
            w.writeheader()
            for e in events:
                w.writerow(
                    {
                        "t_ns": int(e.t_ns),
                        "nonce": int(e.nonce),
                        "hash_hex": str(e.hash_hex),
                        "difficulty_bits": int(e.difficulty_bits),
                        "attempts_since_prev": int(e.attempts_since_prev),
                        "backend": str(e.backend),
                        "notes_json": json.dumps(dict(e.notes), sort_keys=True),
                    }
                )

    def _write_attempt_rate_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_ns", "attempts_total", "attempts_per_s"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ["t_ns", "attempts_total", "attempts_per_s"]})
