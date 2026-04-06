# 03 — Experiments

Supported experiments:

## PC‑CHRONOS readiness v0.2

One-command reproduction:

- `chimera_handoff/scripts/run_pc_chronos_v0_2.sh`

What it runs:

- `condition=steady`
- `condition=heartbeat` at 2.4 Hz and 3.7 Hz (holdout frequency)
- `condition=batching` (200 ms flush)
- `condition=jitter` (±2 ms)
- Seeds `10–19`, difficulty bits `17`
- Summaries via `chimera-summarize`
- DiD decisions via `chimera-did` using decision rule `v0_2_psd_primary`

Key outputs:

- Run roots contain `protocol.json`, `preregistered_metrics.json`, `manifest.json`, and `summary_prng/REPORT.md`.
- DiD outputs contain `REPORT.md` with `DECISION: PASS/FAIL`.

## Thermal stream v1.2 (optional)

- Run: `chimera_handoff/scripts/run_thermal_v1_2.sh`
- Requires usable thermal sensors under `/sys` (script skips if missing).

