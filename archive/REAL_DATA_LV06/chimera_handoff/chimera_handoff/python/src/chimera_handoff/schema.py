from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "0.1"

# Core CSV contracts (PC-CHRONOS).
EVENTS_COLUMNS = ["t_ns", "nonce", "hash_hex", "difficulty_bits", "attempts_since_prev", "backend", "notes_json"]
DELTAS_COLUMNS = ["t_ns", "delta_s"]
CHRONOS_METRICS_COLUMNS = [
    "window_idx",
    "window_t0_ns",
    "window_t1_ns",
    "cv",
    "hist_entropy",
    "hamming_norm",
    "event_rate_hz",
    "psd_peak_hz",
    "psd_peak_power",
    "psd_peak_snr_db",
    "psd_peak_bandwidth_hz",
    "psd_peak_q",
    "psd_peak_hz_error_hz",
]


@dataclass(frozen=True)
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ValidationError(f"missing required file: {path}")


def _require_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise ValidationError(f"missing required dir: {path}")


def _check_csv_header(path: Path, *, expected: List[str]) -> None:
    _require_file(path)
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    if not head:
        raise ValidationError(f"empty CSV: {path}")
    cols = [c.strip() for c in head[0].split(",")]
    if cols != expected:
        raise ValidationError(f"bad CSV header in {path}: expected={expected} got={cols}")


def _iter_run_dirs(run_root: Path) -> Iterable[Tuple[str, Path]]:
    runs = run_root / "runs"
    _require_dir(runs)
    for source_dir in sorted(runs.iterdir()):
        if not source_dir.is_dir():
            continue
        for seed_dir in sorted(source_dir.glob("seed_*")):
            if seed_dir.is_dir():
                yield source_dir.name, seed_dir


def validate_run_root(path: Path) -> None:
    """
    Validates the paper-grade schema contract for a run root produced by the CHIMERA handoff tools.
    Raises ValidationError on violations.
    """
    root = Path(path)
    _require_dir(root)
    _require_file(root / "protocol.json")
    _require_file(root / "manifest.json")
    _require_file(root / "MANIFEST.sha256")
    _require_file(root / "preregistered_metrics.json")

    proto = _read_json(root / "protocol.json")
    if str(proto.get("schema_version")) != SCHEMA_VERSION:
        raise ValidationError(f"bad protocol.schema_version: expected={SCHEMA_VERSION} got={proto.get('schema_version')}")

    any_run = False
    for _, rd in _iter_run_dirs(root):
        any_run = True
        _require_file(rd / "config.json")
        _require_file(rd / "metrics.json")
        # Chronos artifacts are always present for PC-CHRONOS.
        _check_csv_header(rd / "events.csv", expected=EVENTS_COLUMNS)
        _check_csv_header(rd / "deltas.csv", expected=DELTAS_COLUMNS)
        _check_csv_header(rd / "chronos_metrics.csv", expected=CHRONOS_METRICS_COLUMNS)
        _require_file(rd / "chronos_metrics_meta.json")
        _require_file(rd / "chronos_summary.json")
    if not any_run:
        raise ValidationError(f"no runs found under: {root / 'runs'}")

