from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera_handoff.schema import (
    CHRONOS_METRICS_COLUMNS,
    DELTAS_COLUMNS,
    EVENTS_COLUMNS,
    ValidationError,
    validate_run_root,
)


def _write_csv(path: Path, cols: list[str]) -> None:
    path.write_text(",".join(cols) + "\n", encoding="utf-8")


def test_schema_validation_accepts_minimal_run_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "runs" / "prng" / "seed_10").mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"schema_version": "0.1"}) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"schema_version": "0.1"}) + "\n", encoding="utf-8")
    (root / "MANIFEST.sha256").write_text("", encoding="utf-8")
    (root / "preregistered_metrics.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "config.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "metrics.json").write_text("{}", encoding="utf-8")
    _write_csv(root / "runs" / "prng" / "seed_10" / "events.csv", EVENTS_COLUMNS)
    _write_csv(root / "runs" / "prng" / "seed_10" / "deltas.csv", DELTAS_COLUMNS)
    _write_csv(root / "runs" / "prng" / "seed_10" / "chronos_metrics.csv", CHRONOS_METRICS_COLUMNS)
    (root / "runs" / "prng" / "seed_10" / "chronos_metrics_meta.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "chronos_summary.json").write_text("{}", encoding="utf-8")

    validate_run_root(root)


def test_schema_validation_rejects_bad_headers(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "runs" / "prng" / "seed_10").mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"schema_version": "0.1"}) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"schema_version": "0.1"}) + "\n", encoding="utf-8")
    (root / "MANIFEST.sha256").write_text("", encoding="utf-8")
    (root / "preregistered_metrics.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "config.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "metrics.json").write_text("{}", encoding="utf-8")
    _write_csv(root / "runs" / "prng" / "seed_10" / "events.csv", ["t_ns"])  # wrong
    _write_csv(root / "runs" / "prng" / "seed_10" / "deltas.csv", DELTAS_COLUMNS)
    _write_csv(root / "runs" / "prng" / "seed_10" / "chronos_metrics.csv", CHRONOS_METRICS_COLUMNS)
    (root / "runs" / "prng" / "seed_10" / "chronos_metrics_meta.json").write_text("{}", encoding="utf-8")
    (root / "runs" / "prng" / "seed_10" / "chronos_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        validate_run_root(root)

