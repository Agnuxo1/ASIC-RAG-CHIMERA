# Difference-in-Differences (seed paired)

- Idle: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_steady_bits17`
- Burn: `WIP/eigen/reservoir_runs/20251223_pc_chronos_v0_2_heartbeat_3p7_bits17`
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
| psd_peak_hz_error_hz_mean | 0.285592 [0.09802, 0.467008] (d=0.894) | -1.1024 [-1.36966, -0.823479] (d=-2.37) |
| psd_peak_snr_db_mean | -0.0174185 [-0.123399, 0.0802994] (d=-0.0994) | 3.60967 [3.40472, 3.81351] (d=10.3) |
| psd_peak_q_mean | 7.57492 [3.40476, 11.8983] (d=1.04) | -29.2016 [-42.9068, -15.4323] (d=-1.25) |
| psd_peak_hz_iqr_hz | -0.438483 [-1.01778, 0.136716] (d=-0.441) | -1.49365 [-2.25979, -0.809787] (d=-1.21) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.388 [-1.74547, -0.972483] (d=-2.07) | True |
| psd_peak_snr_db_mean | 3.62709 [3.35679, 3.88888] (d=7.9) | True |
| psd_peak_q_mean | -36.7765 [-50.6586, -22.0113] (d=-1.5) | True |
| psd_peak_hz_iqr_hz | -1.05517 [-1.83811, -0.308013] (d=-0.81) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.135994 [-0.0576589, 0.345148] (d=0.392) | -0.390213 [-0.531158, -0.250479] (d=-1.63) |
| psd_peak_snr_db_mean | -0.0149749 [-0.144823, 0.124316] (d=-0.0649) | 1.30648 [0.784869, 1.79086] (d=1.52) |
| psd_peak_q_mean | 2.66044 [-1.27742, 6.80358] (d=0.384) | -19.3127 [-27.2572, -12.3278] (d=-1.5) |
| psd_peak_hz_iqr_hz | -0.386569 [-1.09971, 0.252542] (d=-0.333) | -0.146879 [-0.417028, -0.00905559] (d=-0.346) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.526206 [-0.838035, -0.233837] (d=-1.02) | True |
| psd_peak_snr_db_mean | 1.32146 [0.827259, 1.79814] (d=1.6) | True |
| psd_peak_q_mean | -21.9731 [-28.1545, -16.5206] (d=-2.19) | True |
| psd_peak_hz_iqr_hz | 0.23969 [-0.501083, 1.04414] (d=0.182) | False |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.8327 [0.681912, 1.01835] (d=2.9) | -3.08642 [-3.21124, -2.94663] (d=-13.5) |
| psd_peak_snr_db_mean | -0.605542 [-0.829185, -0.36549] (d=-1.52) | 4.18036 [3.95753, 4.42861] (d=10.3) |
| psd_peak_q_mean | 36.1563 [29.9081, 42.7349] (d=3.29) | 8.61241 [-13.8506, 32.3785] (d=0.218) |
| psd_peak_hz_iqr_hz | 0.31331 [-0.70927, 1.49061] (d=0.166) | -4.93513 [-5.98466, -3.68617] (d=-2.47) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -3.91912 [-4.1863, -3.70601] (d=-9.54) | True |
| psd_peak_snr_db_mean | 4.7859 [4.5016, 5.07789] (d=9.72) | True |
| psd_peak_q_mean | -27.5439 [-49.3618, -2.46913] (d=-0.683) | True |
| psd_peak_hz_iqr_hz | -5.24844 [-6.76558, -3.47171] (d=-1.84) | True |

- DiD metrics with CI excluding 0: 4/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.133468 [-0.148846, 0.40667] (d=0.282) | -1.46582 [-1.83569, -1.11693] (d=-2.37) |
| psd_peak_snr_db_mean | -0.0187339 [-0.149644, 0.107162] (d=-0.0852) | 3.71864 [3.54715, 3.86738] (d=13.6) |
| psd_peak_q_mean | 1.26201 [-4.05787, 6.44401] (d=0.14) | -30.3346 [-48.8359, -11.9822] (d=-0.957) |
| psd_peak_hz_iqr_hz | -0.41847 [-0.943661, 0.139959] (d=-0.45) | -1.85565 [-2.97388, -0.842591] (d=-1.02) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.59929 [-2.04541, -1.16116] (d=-2.11) | True |
| psd_peak_snr_db_mean | 3.73738 [3.60687, 3.87807] (d=16) | True |
| psd_peak_q_mean | -31.5966 [-52.1519, -11.3887] (d=-0.894) | True |
| psd_peak_hz_iqr_hz | -1.43718 [-2.46608, -0.455506] (d=-0.833) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: PASS

