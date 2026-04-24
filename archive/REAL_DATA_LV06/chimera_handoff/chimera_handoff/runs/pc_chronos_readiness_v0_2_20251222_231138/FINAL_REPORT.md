# PC-CHRONOS Readiness v0.2 (handoff) — Final Report

## Run roots (seeds 0-1, duration 20.0s)

- Steady: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_steady_bits17`
- Heartbeat 2.4 Hz: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_heartbeat_2p4_bits17`
- Heartbeat 3.7 Hz: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_heartbeat_3p7_bits17`
- Batching confound (200 ms flush): `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_batching_200ms_bits17`
- Jitter confound (±2 ms): `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_jitter_2ms_bits17`

## Decision artifacts (v0.2 PSD-primary)

All DiD roots below use:
- treatment source: `pc_pow_share_events_metric`
- surrogates: `pc_pow_surrogate_shuffle_metric,pc_pow_surrogate_blockshuffle_metric,pc_pow_surrogate_phase_metric,pc_pow_surrogate_iaaft_metric`
- metrics: `psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz`
- decision rule: `v0_2_psd_primary`

- Heartbeat 2.4 vs steady: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/did_heartbeat_2p4_vs_steady/REPORT.md`
- Heartbeat 3.7 vs steady: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/did_heartbeat_3p7_vs_steady/REPORT.md`
- Batching vs steady: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/did_batching_vs_steady/REPORT.md`
- Jitter vs steady: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/did_jitter_vs_steady/REPORT.md`
