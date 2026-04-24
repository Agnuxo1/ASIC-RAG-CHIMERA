from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def measure_entropy_fingerprint(
    source: object,
    *,
    duration_sec: float = 2.0,
    window_bytes: int = 4096,
) -> Dict[str, Any]:
    """
    Experimental provenance signature: store *only* derived, non-reconstructive summaries.
    Never store raw sampled bytes.
    """
    import time

    t0 = time.perf_counter()
    h = hashlib.sha256()
    total = 0
    windows = 0
    while (time.perf_counter() - t0) < float(duration_sec):
        if hasattr(source, "read_floats"):
            x = np.asarray(getattr(source, "read_floats")(max(16, int(window_bytes) // 4)), dtype=np.float32).reshape(-1)
            if x.size == 0:
                break
            h.update(x.tobytes())
            total += int(x.size)
            windows += 1
        else:
            b = getattr(source, "read_bytes")(int(window_bytes))
            if not b:
                break
            h.update(b)
            total += len(b)
            windows += 1
    return {
        "source_id": getattr(source, "id", "unknown"),
        "stream_kind": "float32" if hasattr(source, "read_floats") else "bytes",
        "duration_sec": float(duration_sec),
        "window_bytes": int(window_bytes),
        "windows": int(windows),
        "bytes_total": int(total) if not hasattr(source, "read_floats") else 0,
        "floats_total": int(total) if hasattr(source, "read_floats") else 0,
        "sha256_of_stream": h.hexdigest(),
        "label": "experimental_provenance_signature_not_security_fingerprint",
    }


def write_entropy_fingerprint(path: Path, fp: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
