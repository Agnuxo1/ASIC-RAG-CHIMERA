from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _busy_loop(stop: threading.Event, *, duty: float, period_s: float) -> None:
    duty = float(max(0.0, min(1.0, duty)))
    period_s = float(max(0.01, period_s))
    t0 = time.perf_counter()
    x = 0.123456789
    while not stop.is_set():
        # Duty-cycled burn.
        phase = (time.perf_counter() - t0) % period_s
        if phase <= duty * period_s:
            # Burn.
            for _ in range(2000):
                x = math.sin(x) + math.cos(x * 1.000001)
        else:
            time.sleep(min(0.005, period_s * 0.05))


@dataclass
class InterventionController:
    mode: str = "none"  # none|cpu_burn|cpu_duty
    threads: int = 1
    duty: float = 1.0
    period_s: float = 1.0
    warmup_s: float = 0.0

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started_at_s: Optional[float] = None

    def start(self) -> Dict[str, Any]:
        m = str(self.mode).lower().strip()
        if m in {"none", ""}:
            self._started_at_s = time.time()
            return {"mode": "none", "started_at_s": float(self._started_at_s)}

        n = int(max(1, self.threads))
        duty = 1.0 if m == "cpu_burn" else float(self.duty)
        period_s = float(self.period_s)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=_busy_loop, args=(self._stop,), kwargs={"duty": duty, "period_s": period_s}, daemon=True)
            for _ in range(n)
        ]
        for t in self._threads:
            t.start()
        self._started_at_s = time.time()
        if float(self.warmup_s) > 0:
            time.sleep(float(self.warmup_s))
        return {
            "mode": str(m),
            "threads": int(n),
            "duty": float(duty),
            "period_s": float(period_s),
            "warmup_s": float(self.warmup_s),
            "started_at_s": float(self._started_at_s),
        }

    def stop(self, *, timeout_s: float = 2.0) -> Dict[str, Any]:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=float(timeout_s))
        stopped_at_s = time.time()
        return {"stopped_at_s": float(stopped_at_s), "n_threads": int(len(self._threads))}

    def __enter__(self) -> InterventionController:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

