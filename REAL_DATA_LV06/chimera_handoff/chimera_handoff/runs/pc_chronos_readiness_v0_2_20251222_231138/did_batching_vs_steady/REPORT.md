# Difference-in-Differences (seed paired)

- Idle: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_steady_bits17`
- Burn: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_batching_200ms_bits17`
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
| did | k-of-n CI excludes 0 | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_blockshuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| v0.2 | overall |  | False |

## Confound Gates (v0.2)

- condition: `batching`
- C0_no_target_lock_pass: `True` (tol_hz=0.15, median_err=5.424853573905588)
- batching_expected_hz: `5.0`
- psd_peak_hz_median_median: `10.0`
- psd_peak_hz_error_to_batching_hz_median: `5.0`

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.282347 [-1.18681, 0.622112] (d=-0.221) | 1.48703 [1.32947, 1.64458] (d=6.67) |
| psd_peak_snr_db_mean | -0.914165 [-1.49249, -0.33584] (d=-1.12) | 0.524893 [0.149821, 0.899966] (d=0.99) |
| psd_peak_q_mean | -0.676101 [-0.935535, -0.416667] (d=-1.84) | 8.31965 [4.57407, 12.0652] (d=1.57) |
| psd_peak_hz_iqr_hz | 0.873808 [0.173433, 1.57418] (d=0.882) | 1.64463 [-0.0409917, 3.33025] (d=0.69) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 1.76937 [0.707357, 2.83139] (d=1.18) | True |
| psd_peak_snr_db_mean | 1.43906 [1.23581, 1.64231] (d=5.01) | True |
| psd_peak_q_mean | 8.99575 [5.50961, 12.4819] (d=1.82) | True |
| psd_peak_hz_iqr_hz | 0.770821 [-1.61518, 3.15682] (d=0.228) | False |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.514604 [-0.169238, 1.19845] (d=0.532) | 2.27103 [1.51004, 3.03201] (d=2.11) |
| psd_peak_snr_db_mean | -1.72351 [-3.12255, -0.324466] (d=-0.871) | 0.547644 [-0.424702, 1.51999] (d=0.398) |
| psd_peak_q_mean | 1.83798 [-1.23913, 4.91509] (d=0.422) | 17.2886 [13.3056, 21.2717] (d=3.07) |
| psd_peak_hz_iqr_hz | 0.528741 [-0.440368, 1.49785] (d=0.386) | 2.13887 [-0.738278, 5.01602] (d=0.526) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 1.75642 [0.311594, 3.20125] (d=0.86) | True |
| psd_peak_snr_db_mean | 2.27115 [1.84446, 2.69784] (d=3.76) | True |
| psd_peak_q_mean | 15.4507 [8.39046, 22.5109] (d=1.55) | True |
| psd_peak_hz_iqr_hz | 1.61013 [-2.23613, 5.45639] (d=0.296) | False |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0860293 [-0.55338, 0.381322] (d=-0.13) | 3.58758 [3.43822, 3.73694] (d=17) |
| psd_peak_snr_db_mean | -1.41362 [-1.72394, -1.1033] (d=-3.22) | -11.1847 [-11.7439, -10.6255] (d=-14.1) |
| psd_peak_q_mean | 2.16868 [-0.216981, 4.55435] (d=0.643) | 31.6902 [26.4681, 36.9123] (d=4.29) |
| psd_peak_hz_iqr_hz | -2.26847 [-3.89961, -0.637325] (d=-0.983) | 3.78125 [3.33333, 4.22916] (d=5.97) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 3.67361 [3.35562, 3.9916] (d=8.17) | True |
| psd_peak_snr_db_mean | -9.77111 [-10.6407, -8.90157] (d=-7.95) | True |
| psd_peak_q_mean | 29.5215 [26.6851, 32.3579] (d=7.36) | True |
| psd_peak_hz_iqr_hz | 6.04971 [3.97066, 8.12877] (d=2.06) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.750064 [0.508762, 0.991367] (d=2.2) | 2.20765 [1.99656, 2.41875] (d=7.39) |
| psd_peak_snr_db_mean | -0.932553 [-1.11094, -0.75417] (d=-3.7) | 0.666327 [0.365094, 0.967561] (d=1.56) |
| psd_peak_q_mean | 4.41737 [1.55031, 7.28442] (d=1.09) | 11.6341 [7.83333, 15.4348] (d=2.16) |
| psd_peak_hz_iqr_hz | 0.59436 [0.264653, 0.924066] (d=1.27) | 4.88046 [4.76804, 4.99288] (d=30.7) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 1.45759 [1.00519, 1.90998] (d=2.28) | True |
| psd_peak_snr_db_mean | 1.59888 [1.47603, 1.72173] (d=9.2) | True |
| psd_peak_q_mean | 7.21669 [6.28302, 8.15036] (d=5.47) | True |
| psd_peak_hz_iqr_hz | 4.2861 [3.84397, 4.72822] (d=6.85) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: FAIL

