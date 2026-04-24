# 05 — Surrogates

Surrogates are used to falsify “structure” claims by comparing the treated stream against destroyed-structure controls.

Implemented (for floating series):

- `shuffle`: random permutation.
- `blockshuffle`: permute contiguous blocks (preserves local structure within block).
- `phase`: FFT magnitude preserved, phases randomized (destroys temporal structure).
- `iaaft`: iterative amplitude-adjusted FFT (matches amplitude distribution + approx spectrum).

Determinism:

- All surrogates are deterministic given `seed` (and, when used in pipelines, the window index is incorporated upstream).

Safety fix (span control for log‑delta surrogates):

- When surrogates are applied to `log10(delta_s)`, reconstructed `delta_s` are clipped and renormalized to preserve total span (prevents pathological “blow up” in duration that would destabilize PSD binning).

