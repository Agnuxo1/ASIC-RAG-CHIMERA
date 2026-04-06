from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class EventRecord:
    t_ns: int
    nonce: int
    hash_hex: str
    difficulty_bits: int
    attempts_since_prev: int
    backend: str = "cpu"  # cpu|gpu
    notes: Dict[str, Any] = field(default_factory=dict)

