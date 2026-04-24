# Difference-in-Differences (seed paired)

- Idle: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_steady_bits17`
- Burn: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_batching_200ms_bits17`
- Metrics: `psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz`
- Surrogates: `pc_pow_surrogate_shuffle_metric,pc_pow_surrogate_blockshuffle_metric,pc_pow_surrogate_phase_metric,pc_pow_surrogate_iaaft_metric`

## Preregistration

- primary_metrics: `psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz`
- primary_effect_threshold: `None`
- replication_rule: k=3 of n=4

## Decision Summary

| category | condition | passing_surrogates | pass_requires_iaaft_and_2_surrogates |
|---|---|---|---:|
| within | idle | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric | False |
| within | burn | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_blockshuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| did | k-of-n same-direction |  | False |
| did | k-of-n CI excludes 0 | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| v0.2 | overall |  | False |

## Confound Gates (v0.2)

- condition: `batching`
- C0_no_target_lock_pass: `True` (tol_hz=0.15, median_err=4.1581557772359705)
- batching_expected_hz: `5.0`
- psd_peak_hz_median_median: `5.0`
- psd_peak_hz_error_to_batching_hz_median: `0.0`

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.285592 [0.09802, 0.467008] (d=0.894) | 1.16701 [1.01886, 1.32996] (d=4.34) |
| psd_peak_snr_db_mean | -0.0174185 [-0.123399, 0.0802994] (d=-0.0994) | 1.66684 [1.39568, 1.90994] (d=3.76) |
| psd_peak_q_mean | 7.57492 [3.40476, 11.8983] (d=1.04) | 19.3369 [16.8181, 21.7662] (d=4.53) |
| psd_peak_hz_iqr_hz | -0.438483 [-1.01778, 0.136716] (d=-0.441) | 4.99916 [4.99861, 4.9997] (d=5.35e+03) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.881415 [0.580072, 1.20032] (d=1.65) | True |
| psd_peak_snr_db_mean | 1.68426 [1.41034, 1.93483] (d=3.77) | True |
| psd_peak_q_mean | 11.762 [6.64865, 16.8162] (d=1.35) | True |
| psd_peak_hz_iqr_hz | 5.43764 [4.86224, 6.01741] (d=5.47) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_snr_db_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.135994 [-0.0576589, 0.345148] (d=0.392) | 0.943915 [-0.054206, 1.54201] (d=0.623) |
| psd_peak_snr_db_mean | -0.0149749 [-0.144823, 0.124316] (d=-0.0649) | 0.859736 [0.44479, 1.19246] (d=1.34) |
| psd_peak_q_mean | 2.66044 [-1.27742, 6.80358] (d=0.384) | 17.5541 [-5.82398, 33.1886] (d=0.5) |
| psd_peak_hz_iqr_hz | -0.386569 [-1.09971, 0.252542] (d=-0.333) | 5.00647 [5.00362, 5.00905] (d=1.08e+03) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.807922 [-0.161886, 1.4549] (d=0.552) | False |
| psd_peak_snr_db_mean | 0.874711 [0.407637, 1.28197] (d=1.18) | True |
| psd_peak_q_mean | 14.8937 [-6.61248, 30.5991] (d=0.457) | False |
| psd_peak_hz_iqr_hz | 5.39303 [4.75636, 6.10391] (d=4.67) | True |

- DiD metrics with CI excluding 0: 2/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.8327 [0.681912, 1.01835] (d=2.9) | 1.75899 [1.51062, 1.98617] (d=4.31) |
| psd_peak_snr_db_mean | -0.605542 [-0.829185, -0.36549] (d=-1.52) | 6.08577 [1.20838, 10.3668] (d=0.779) |
| psd_peak_q_mean | 36.1563 [29.9081, 42.7349] (d=3.29) | 79.1332 [62.6469, 88.3679] (d=3.14) |
| psd_peak_hz_iqr_hz | 0.31331 [-0.70927, 1.49061] (d=0.166) | 3.96881 [3.09536, 4.65412] (d=2.95) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.926285 [0.616915, 1.20506] (d=1.84) | True |
| psd_peak_snr_db_mean | 6.69131 [1.76115, 10.9909] (d=0.848) | True |
| psd_peak_q_mean | 42.9768 [20.4867, 57.5438] (d=1.28) | True |
| psd_peak_hz_iqr_hz | 3.6555 [2.15254, 4.97271] (d=1.51) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.133468 [-0.148846, 0.40667] (d=0.282) | 1.06798 [0.915963, 1.21629] (d=4.16) |
| psd_peak_snr_db_mean | -0.0187339 [-0.149644, 0.107162] (d=-0.0852) | 1.26496 [1.02543, 1.53608] (d=2.86) |
| psd_peak_q_mean | 1.26201 [-4.05787, 6.44401] (d=0.14) | 17.4748 [14.3372, 19.9684] (d=3.56) |
| psd_peak_hz_iqr_hz | -0.41847 [-0.943661, 0.139959] (d=-0.45) | 4.99393 [4.98328, 4.99957] (d=301) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.934511 [0.627708, 1.23787] (d=1.8) | True |
| psd_peak_snr_db_mean | 1.28369 [1.03986, 1.52458] (d=3.09) | True |
| psd_peak_q_mean | 16.2128 [9.47458, 22.6787] (d=1.43) | True |
| psd_peak_hz_iqr_hz | 5.4124 [4.84816, 5.94149] (d=5.77) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: FAIL

