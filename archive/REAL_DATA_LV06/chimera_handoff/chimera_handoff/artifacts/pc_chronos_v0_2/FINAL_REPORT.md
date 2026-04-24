# PC‑CHRONOS Readiness v0.2 (PSD‑Primary) — Final Report

This report executes the rerun requested in `WIP/eigen/eigen_final_scic_extended.md` and the frozen spec in `WIP/eigen/eigen_pc_chronos_v0_2.md`.

v0.2 differs from v0.1 by making **PSD metrics primary** (decision‑gating) and requiring a rerun on **fresh seeds** plus a **holdout heartbeat frequency**.

## Run roots (fresh seeds 10–19)

- Steady: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_steady_bits17`
- Heartbeat 2.4 Hz: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_heartbeat_2p4_bits17`
- Heartbeat holdout 3.7 Hz: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_heartbeat_3p7_bits17`
- Batching confound (200 ms flush): `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_batching_200ms_bits17`
- Jitter confound (±2 ms): `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_jitter_2ms_bits17`

## Decision artifacts (v0.2 PSD‑primary)

All DiD roots below use:
- treatment source: `pc_pow_share_events_metric`
- surrogates: `pc_pow_surrogate_{shuffle,blockshuffle,phase,iaaft}_metric`
- metrics: `psd_peak_hz_error_hz_mean, psd_peak_snr_db_mean, psd_peak_q_mean, psd_peak_hz_iqr_hz`
- decision rule: `v0_2_psd_primary`

- Heartbeat 2.4 vs steady: `WIP/eigen/reservoir_runs/20251223_pc_chronos_readiness_v0_2/did_heartbeat_2p4_vs_steady/REPORT.md`
- Heartbeat 3.7 vs steady: `WIP/eigen/reservoir_runs/20251223_pc_chronos_readiness_v0_2/did_heartbeat_3p7_vs_steady/REPORT.md`
- Batching vs steady: `WIP/eigen/reservoir_runs/20251223_pc_chronos_readiness_v0_2/did_batching_vs_steady/REPORT.md`
- Jitter vs steady: `WIP/eigen/reservoir_runs/20251223_pc_chronos_readiness_v0_2/did_jitter_vs_steady/REPORT.md`

## Primary outcomes (PSD‑primary go/no‑go)

### Heartbeat detection (PASS at both frequencies)

Both heartbeat frequencies satisfy the v0.2 “PSD‑primary DiD vs surrogates” gate:

- `did_heartbeat_2p4_vs_steady/REPORT.md` ends with `DECISION: PASS`
- `did_heartbeat_3p7_vs_steady/REPORT.md` ends with `DECISION: PASS`

Summary sanity (treated stream `pc_pow_share_events_metric`, medians over seeds 10–19):

- Steady: peak ≈ 5.35 Hz, error-to-2.4 ≈ 3.30 Hz
- Heartbeat 2.4: peak ≈ 2.3987 Hz, error-to-2.4 ≈ 0.0042 Hz
- Heartbeat 3.7: peak ≈ 3.6981 Hz, error-to-3.7 ≈ 0.0041 Hz

### Confound rejection (PASS: no false target lock)

Confound gates (C0) are explicitly reported in:

- `did_batching_vs_steady/REPORT.md` (C0 pass; batching peak explained at 5 Hz)
- `did_jitter_vs_steady/REPORT.md` (C0 pass; error-to-target median far above tolerance)

Batching explanation check (C1):
- batching expected peak = 5.0 Hz (from 200 ms flush)
- observed peak median = 5.0 Hz

## Bottom line (ASIC readiness meaning)

v0.2 meets the intended “go/no‑go” criteria:

1. Heartbeat @ 2.4 Hz passes the PSD‑primary DiD gate.
2. Heartbeat @ holdout 3.7 Hz also passes.
3. Batching and jitter do not false‑lock to the target heartbeat frequency and are correctly identifiable as confounds.

Under the PC‑only proxy, the Chronos timing observable + PSD extraction + surrogate falsification harness is now decision‑grade for proceeding to ASIC ingestion with the same analysis stack.

