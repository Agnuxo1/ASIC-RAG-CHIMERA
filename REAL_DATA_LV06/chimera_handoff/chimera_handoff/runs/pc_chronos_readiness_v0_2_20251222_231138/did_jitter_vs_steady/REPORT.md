# Difference-in-Differences (seed paired)

- Idle: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_steady_bits17`
- Burn: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_jitter_2ms_bits17`
- Metrics: `psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz`
- Surrogates: `pc_pow_surrogate_shuffle_metric,pc_pow_surrogate_blockshuffle_metric,pc_pow_surrogate_phase_metric,pc_pow_surrogate_iaaft_metric`

## Preregistration

- primary_metrics: `psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz`
- primary_effect_threshold: `None`
- replication_rule: k=3 of n=4

## Decision Summary

| category | condition | passing_surrogates | pass_requires_iaaft_and_2_surrogates |
|---|---|---|---:|
| within | idle | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| within | burn | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_blockshuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| did | k-of-n same-direction |  | False |
| did | k-of-n CI excludes 0 | pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| v0.2 | overall | pc_pow_surrogate_iaaft_metric | False |

## Confound Gates (v0.2)

- condition: `jitter`
- C0_no_target_lock_pass: `True` (tol_hz=0.15, median_err=3.067762829760368)

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.282347 [-1.18681, 0.622112] (d=-0.221) | -0.6089 [-0.684712, -0.533087] (d=-5.68) |
| psd_peak_snr_db_mean | -0.914165 [-1.49249, -0.33584] (d=-1.12) | -0.73626 [-1.57678, 0.104259] (d=-0.619) |
| psd_peak_q_mean | -0.676101 [-0.935535, -0.416667] (d=-1.84) | -3.32816 [-5.97484, -0.681481] (d=-0.889) |
| psd_peak_hz_iqr_hz | 0.873808 [0.173433, 1.57418] (d=0.882) | -1.13364 [-1.13998, -1.1273] (d=-126) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.326552 [-1.30682, 0.65372] (d=-0.236) | False |
| psd_peak_snr_db_mean | 0.177905 [-0.0842895, 0.440099] (d=0.48) | False |
| psd_peak_q_mean | -2.65206 [-5.03931, -0.264815] (d=-0.786) | True |
| psd_peak_hz_iqr_hz | -2.00745 [-2.70148, -1.31341] (d=-2.05) | True |

- DiD metrics with CI excluding 0: 2/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.514604 [-0.169238, 1.19845] (d=0.532) | 0.526672 [0.484164, 0.56918] (d=8.76) |
| psd_peak_snr_db_mean | -1.72351 [-3.12255, -0.324466] (d=-0.871) | -1.55896 [-2.15564, -0.962271] (d=-1.85) |
| psd_peak_q_mean | 1.83798 [-1.23913, 4.91509] (d=0.422) | 2.94586 [2.88889, 3.00283] (d=36.6) |
| psd_peak_hz_iqr_hz | 0.528741 [-0.440368, 1.49785] (d=0.386) | -0.00791392 [-2.20473, 2.1889] (d=-0.00255) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.0120682 [-0.629266, 0.653403] (d=0.0133) | False |
| psd_peak_snr_db_mean | 0.164549 [-0.637804, 0.966902] (d=0.145) | False |
| psd_peak_q_mean | 1.10788 [-1.91226, 4.12802] (d=0.259) | False |
| psd_peak_hz_iqr_hz | -0.536655 [-1.76436, 0.691051] (d=-0.309) | False |

- DiD metrics with CI excluding 0: 0/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0860293 [-0.55338, 0.381322] (d=-0.13) | 0.639063 [0.0983712, 1.17975] (d=0.836) |
| psd_peak_snr_db_mean | -1.41362 [-1.72394, -1.1033] (d=-3.22) | -0.425086 [-0.530856, -0.319315] (d=-2.84) |
| psd_peak_q_mean | 2.16868 [-0.216981, 4.55435] (d=0.643) | 6.9127 [4.69815, 9.12725] (d=2.21) |
| psd_peak_hz_iqr_hz | -2.26847 [-3.89961, -0.637325] (d=-0.983) | -0.447943 [-3.11944, 2.22356] (d=-0.119) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.725092 [0.651751, 0.798433] (d=6.99) | True |
| psd_peak_snr_db_mean | 0.988533 [0.572442, 1.40462] (d=1.68) | True |
| psd_peak_q_mean | 4.74402 [0.1438, 9.34423] (d=0.729) | True |
| psd_peak_hz_iqr_hz | 1.82052 [0.780163, 2.86088] (d=1.24) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.750064 [0.508762, 0.991367] (d=2.2) | -0.490722 [-0.766512, -0.214932] (d=-1.26) |
| psd_peak_snr_db_mean | -0.932553 [-1.11094, -0.75417] (d=-3.7) | -0.498914 [-0.79573, -0.202097] (d=-1.19) |
| psd_peak_q_mean | 4.41737 [1.55031, 7.28442] (d=1.09) | -2.48 [-2.92296, -2.03704] (d=-3.96) |
| psd_peak_hz_iqr_hz | 0.59436 [0.264653, 0.924066] (d=1.27) | -1.57645 [-1.91493, -1.23798] (d=-3.29) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.24079 [-1.75788, -0.723694] (d=-1.7) | True |
| psd_peak_snr_db_mean | 0.433639 [0.315205, 0.552073] (d=2.59) | True |
| psd_peak_q_mean | -6.89736 [-9.32146, -4.47327] (d=-2.01) | True |
| psd_peak_hz_iqr_hz | -2.17081 [-2.839, -1.50263] (d=-2.3) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: FAIL

