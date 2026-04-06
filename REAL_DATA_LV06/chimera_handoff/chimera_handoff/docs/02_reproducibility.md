# 02 — Reproducibility

## Python

- Dependency pinning: `chimera_handoff/python/requirements.lock` (hash-locked).
- Install path: `chimera_handoff/scripts/setup_python.sh` creates `chimera_handoff/python/.venv`.

Determinism rules:

- All scripts set seeds explicitly and record them in `protocol.json` and `preregistered_metrics.json`.
- Surrogates are deterministic given `(seed, window_idx)` (see `docs/05_surrogates.md`).

## Lean

- Toolchain pin: `chimera_handoff/lean/lean-toolchain`
- Package pin: `chimera_handoff/lean/lake-manifest.json`
- Strict build: `lake build -- -Dno_sorry -DwarningAsError=true`

## Integrity

- Each run root includes `manifest.json` plus `MANIFEST.sha256`.
- Verify locally: `chimera_handoff/scripts/verify_manifests.sh`.

