# 00 — Overview

`chimera_handoff/` is a standalone handoff bundle containing:

- A reproducible Python package for CHIMERA experiments (PC‑CHRONOS + thermal stream) with CLIs, schema validation, manifests, and tests.
- A minimal Lean 4 project containing the nucleus/Heyting core used in the work (pinned toolchain + pinned Mathlib).
- Curated example artifacts (small, paper-grade) for immediate inspection.

## Folder map

- `chimera_handoff/python/`: runnable Python package + tests.
- `chimera_handoff/lean/`: minimal Lean lake project.
- `chimera_handoff/scripts/`: setup, test, reproduction, manifest verification.
- `chimera_handoff/docs/`: technical docs and schema contract.
- `chimera_handoff/artifacts/`: curated outputs (no huge raw dumps).

