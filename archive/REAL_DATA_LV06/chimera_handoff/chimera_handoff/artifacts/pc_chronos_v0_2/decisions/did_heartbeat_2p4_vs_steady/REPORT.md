# Difference-in-Differences (seed paired)

- Idle: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_steady_bits17`
- Burn: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_heartbeat_2p4_bits17`
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
| did | k-of-n CI excludes 0 | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_blockshuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| v0.2 | overall | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.285592 [0.09802, 0.467008] (d=0.894) | -2.14569 [-2.52309, -1.78672] (d=-3.42) |
| psd_peak_snr_db_mean | -0.0174185 [-0.123399, 0.0802994] (d=-0.0994) | 3.94024 [3.58914, 4.26779] (d=6.77) |
| psd_peak_q_mean | 7.57492 [3.40476, 11.8983] (d=1.04) | -73.3082 [-97.5012, -52.3531] (d=-1.89) |
| psd_peak_hz_iqr_hz | -0.438483 [-1.01778, 0.136716] (d=-0.441) | -3.53973 [-4.38737, -2.59598] (d=-2.31) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -2.43128 [-2.82166, -2.06715] (d=-3.76) | True |
| psd_peak_snr_db_mean | 3.95766 [3.5652, 4.3214] (d=6.1) | True |
| psd_peak_q_mean | -80.8831 [-104.549, -59.3004] (d=-2.08) | True |
| psd_peak_hz_iqr_hz | -3.10124 [-4.28603, -1.78595] (d=-1.45) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.135994 [-0.0576589, 0.345148] (d=0.392) | -0.164707 [-0.277443, -0.0723724] (d=-0.925) |
| psd_peak_snr_db_mean | -0.0149749 [-0.144823, 0.124316] (d=-0.0649) | 1.37525 [1.07403, 1.62896] (d=2.91) |
| psd_peak_q_mean | 2.66044 [-1.27742, 6.80358] (d=0.384) | -11.7137 [-20.9921, -4.03732] (d=-0.796) |
| psd_peak_hz_iqr_hz | -0.386569 [-1.09971, 0.252542] (d=-0.333) | -0.0157466 [-0.0258111, -0.00896425] (d=-1.04) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.300701 [-0.54004, -0.0678396] (d=-0.739) | True |
| psd_peak_snr_db_mean | 1.39022 [1.06047, 1.70514] (d=2.51) | True |
| psd_peak_q_mean | -14.3741 [-21.8792, -7.34681] (d=-1.15) | True |
| psd_peak_hz_iqr_hz | 0.370822 [-0.2673, 1.08464] (d=0.319) | False |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.8327 [0.681912, 1.01835] (d=2.9) | -2.36789 [-2.53436, -2.1734] (d=-7.73) |
| psd_peak_snr_db_mean | -0.605542 [-0.829185, -0.36549] (d=-1.52) | 3.71347 [3.36478, 4.05658] (d=6.28) |
| psd_peak_q_mean | 36.1563 [29.9081, 42.7349] (d=3.29) | -13.7164 [-26.873, -0.753276] (d=-0.612) |
| psd_peak_hz_iqr_hz | 0.31331 [-0.70927, 1.49061] (d=0.166) | -3.45634 [-4.19316, -2.71968] (d=-2.78) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -3.20059 [-3.44877, -2.95846] (d=-7.68) | True |
| psd_peak_snr_db_mean | 4.31901 [3.99381, 4.64323] (d=7.8) | True |
| psd_peak_q_mean | -49.8728 [-62.3097, -37.4423] (d=-2.33) | True |
| psd_peak_hz_iqr_hz | -3.76965 [-5.1206, -2.35602] (d=-1.6) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.133468 [-0.148846, 0.40667] (d=0.282) | -1.63384 [-1.91193, -1.32438] (d=-3.23) |
| psd_peak_snr_db_mean | -0.0187339 [-0.149644, 0.107162] (d=-0.0852) | 3.8407 [3.4666, 4.19375] (d=6.12) |
| psd_peak_q_mean | 1.26201 [-4.05787, 6.44401] (d=0.14) | -52.2765 [-67.6738, -39.0849] (d=-2.13) |
| psd_peak_hz_iqr_hz | -0.41847 [-0.943661, 0.139959] (d=-0.45) | -2.3845 [-3.34987, -1.37576] (d=-1.4) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.7673 [-2.22823, -1.22245] (d=-2.06) | True |
| psd_peak_snr_db_mean | 3.85943 [3.49419, 4.24737] (d=5.99) | True |
| psd_peak_q_mean | -53.5385 [-71.0632, -37.221] (d=-1.84) | True |
| psd_peak_hz_iqr_hz | -1.96603 [-3.01325, -0.823332] (d=-1.05) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: PASS

