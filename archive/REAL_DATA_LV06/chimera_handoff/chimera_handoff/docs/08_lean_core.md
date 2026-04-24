# 08 — Lean Core

Lean project root: `chimera_handoff/lean/`

## What is included

Minimal, audit-friendly files:

- `chimera_handoff/lean/HeytingLean/Eigen/NucleusReLU.lean`
- `chimera_handoff/lean/HeytingLean/Eigen/NucleusThreshold.lean`
- `chimera_handoff/lean/HeytingLean/Eigen/ParametricHeyting.lean` (adjunction / residuation: `PBVec.himp_adjoint`)
- `chimera_handoff/lean/HeytingLean/Eigen/ThreeLaws.lean` (structural “bridge” lemmas)

Smoke target:

- `chimera_handoff/lean/Tests/Smoke.lean`

## Build (strict)

```bash
cd chimera_handoff/lean
lake update
lake build -- -Dno_sorry -DwarningAsError=true
```

## Key statements

- Threshold nucleus fixed-point characterization: `HeytingLean.Eigen.thresholdNucleus_fixed_iff`
- Parametric Heyting adjunction: `HeytingLean.Eigen.PBVec.himp_adjoint`

