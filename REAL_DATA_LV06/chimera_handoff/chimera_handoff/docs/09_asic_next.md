# 09 — ASIC Next

PC‑CHRONOS is a PC-only proxy. For ASIC ingestion:

1. Replace the event source:
   - Instead of generating PoW timing internally, ingest Stratum/share logs (or device telemetry) and emit the same `events.csv` schema.
2. Keep the analysis stack unchanged:
   - `events.csv` → `deltas.csv` → `chronos_metrics.csv` → summaries → DiD decisions.
3. Preserve prereg + manifest discipline:
   - emit `protocol.json`, `preregistered_metrics.json`, `manifest.json`, and `MANIFEST.sha256` per run root.

Minimal ingestion target:

- One CSV row per accepted share/event with a strictly increasing `t_ns` timestamp.

