# Difference-in-Differences (seed paired)

- Idle: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_steady_bits17`
- Burn: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_jitter_2ms_bits17`
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
| within | burn | pc_pow_surrogate_phase_metric | False |
| did | k-of-n same-direction |  | False |
| did | k-of-n CI excludes 0 | pc_pow_surrogate_phase_metric | False |
| v0.2 | overall |  | False |

## Confound Gates (v0.2)

- condition: `jitter`
- C0_no_target_lock_pass: `True` (tol_hz=0.15, median_err=3.151580745280997)

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=pos, passed=psd_peak_hz_error_hz_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.285592 [0.09802, 0.467008] (d=0.894) | 0.121904 [-0.113685, 0.354467] (d=0.3) |
| psd_peak_snr_db_mean | -0.0174185 [-0.123399, 0.0802994] (d=-0.0994) | 0.0270753 [-0.126904, 0.175342] (d=0.105) |
| psd_peak_q_mean | 7.57492 [3.40476, 11.8983] (d=1.04) | 3.87772 [0.234226, 7.31358] (d=0.64) |
| psd_peak_hz_iqr_hz | -0.438483 [-1.01778, 0.136716] (d=-0.441) | -0.219465 [-0.802348, 0.284044] (d=-0.235) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.163687 [-0.33453, -0.0117571] (d=-0.593) | True |
| psd_peak_snr_db_mean | 0.0444938 [-0.0891897, 0.179423] (d=0.196) | False |
| psd_peak_q_mean | -3.6972 [-8.14997, 0.952763] (d=-0.474) | False |
| psd_peak_hz_iqr_hz | 0.219017 [-0.468275, 0.933113] (d=0.182) | False |

- DiD metrics with CI excluding 0: 1/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.135994 [-0.0576589, 0.345148] (d=0.392) | 0.0844515 [-0.111256, 0.284464] (d=0.251) |
| psd_peak_snr_db_mean | -0.0149749 [-0.144823, 0.124316] (d=-0.0649) | -0.162373 [-0.331995, -0.0159209] (d=-0.602) |
| psd_peak_q_mean | 2.66044 [-1.27742, 6.80358] (d=0.384) | 1.97042 [-1.45775, 5.28293] (d=0.346) |
| psd_peak_hz_iqr_hz | -0.386569 [-1.09971, 0.252542] (d=-0.333) | -0.249427 [-0.732145, 0.174234] (d=-0.32) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0515422 [-0.229749, 0.103819] (d=-0.179) | False |
| psd_peak_snr_db_mean | -0.147398 [-0.335948, 0.0215023] (d=-0.482) | False |
| psd_peak_q_mean | -0.690019 [-4.7273, 3.04456] (d=-0.104) | False |
| psd_peak_hz_iqr_hz | 0.137141 [-0.397843, 0.791477] (d=0.134) | False |

- DiD metrics with CI excluding 0: 0/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.8327 [0.681912, 1.01835] (d=2.9) | 0.983567 [0.777648, 1.18027] (d=2.84) |
| psd_peak_snr_db_mean | -0.605542 [-0.829185, -0.36549] (d=-1.52) | -1.03753 [-1.42062, -0.692454] (d=-1.67) |
| psd_peak_q_mean | 36.1563 [29.9081, 42.7349] (d=3.29) | 44.3878 [36.5132, 51.2973] (d=3.51) |
| psd_peak_hz_iqr_hz | 0.31331 [-0.70927, 1.49061] (d=0.166) | 2.77257 [1.65467, 3.79722] (d=1.52) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.150867 [-0.0407469, 0.328324] (d=0.471) | False |
| psd_peak_snr_db_mean | -0.431992 [-0.780236, -0.0656686] (d=-0.714) | True |
| psd_peak_q_mean | 8.2315 [1.45381, 14.7769] (d=0.725) | True |
| psd_peak_hz_iqr_hz | 2.45926 [1.17489, 3.71675] (d=1.14) | True |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.133468 [-0.148846, 0.40667] (d=0.282) | 0.0519814 [-0.139057, 0.257218] (d=0.154) |
| psd_peak_snr_db_mean | -0.0187339 [-0.149644, 0.107162] (d=-0.0852) | -0.0498531 [-0.172968, 0.0468897] (d=-0.265) |
| psd_peak_q_mean | 1.26201 [-4.05787, 6.44401] (d=0.14) | 0.986612 [-2.948, 5.0335] (d=0.145) |
| psd_peak_hz_iqr_hz | -0.41847 [-0.943661, 0.139959] (d=-0.45) | -0.0873927 [-0.607991, 0.406656] (d=-0.1) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0814869 [-0.441505, 0.267942] (d=-0.135) | False |
| psd_peak_snr_db_mean | -0.0311191 [-0.17196, 0.113252] (d=-0.127) | False |
| psd_peak_q_mean | -0.275399 [-7.49615, 6.53256] (d=-0.0229) | False |
| psd_peak_hz_iqr_hz | 0.331078 [-0.314993, 0.995705] (d=0.293) | False |

- DiD metrics with CI excluding 0: 0/4

DECISION: FAIL

