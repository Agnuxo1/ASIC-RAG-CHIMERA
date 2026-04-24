# 01 — Quickstart

Prereqs:

- Linux (recommended; thermal stream reads `/sys`).
- `python3` (>= 3.10), `git`, `curl`, `sha256sum`.

```bash
cd chimera_handoff
./scripts/setup_python.sh
./scripts/run_all_tests.sh
./scripts/run_pc_chronos_v0_2.sh
```

Outputs:

- PC‑CHRONOS runs: `chimera_handoff/runs/pc_chronos_readiness_v0_2_<timestamp>/`
- DiD decisions: `.../did_*/REPORT.md` (look for `DECISION: PASS/FAIL`)
