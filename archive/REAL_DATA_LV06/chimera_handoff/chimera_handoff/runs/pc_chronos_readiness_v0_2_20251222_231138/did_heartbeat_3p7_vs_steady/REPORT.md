# Difference-in-Differences (seed paired)

- Idle: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_steady_bits17`
- Burn: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_heartbeat_3p7_bits17`
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
| within | burn | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| did | k-of-n same-direction | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric | False |
| did | k-of-n CI excludes 0 | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |
| v0.2 | overall | pc_pow_surrogate_shuffle_metric, pc_pow_surrogate_phase_metric, pc_pow_surrogate_iaaft_metric | True |

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=True (k=3 of n=4, direction=pos, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.282347 [-1.18681, 0.622112] (d=-0.221) | -1.28388 [-1.44427, -1.12348] (d=-5.66) |
| psd_peak_snr_db_mean | -0.914165 [-1.49249, -0.33584] (d=-1.12) | -0.0980934 [-0.209752, 0.0135655] (d=-0.621) |
| psd_peak_q_mean | -0.676101 [-0.935535, -0.416667] (d=-1.84) | -12.4451 [-37.0449, 12.1548] (d=-0.358) |
| psd_peak_hz_iqr_hz | 0.873808 [0.173433, 1.57418] (d=0.882) | -2.86688 [-3.24726, -2.4865] (d=-5.33) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.00153 [-1.74559, -0.257462] (d=-0.952) | True |
| psd_peak_snr_db_mean | 0.816072 [0.126088, 1.50606] (d=0.836) | True |
| psd_peak_q_mean | -11.769 [-36.6282, 13.0903] (d=-0.335) | False |
| psd_peak_hz_iqr_hz | -3.74069 [-4.06068, -3.42069] (d=-8.27) | True |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.514604 [-0.169238, 1.19845] (d=0.532) | -0.256874 [-0.513748, 0] (d=-0.707) |
| psd_peak_snr_db_mean | -1.72351 [-3.12255, -0.324466] (d=-0.871) | -0.889034 [-1.77807, 0] (d=-0.707) |
| psd_peak_q_mean | 1.83798 [-1.23913, 4.91509] (d=0.422) | -16.5288 [-33.0577, 0] (d=-0.707) |
| psd_peak_hz_iqr_hz | 0.528741 [-0.440368, 1.49785] (d=0.386) | -0.129206 [-0.258412, 0] (d=-0.707) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.771478 [-1.19845, -0.34451] (d=-1.28) | True |
| psd_peak_snr_db_mean | 0.834472 [-1.4536, 3.12255] (d=0.258) | False |
| psd_peak_q_mean | -18.3668 [-31.8186, -4.91509] (d=-0.965) | True |
| psd_peak_hz_iqr_hz | -0.657947 [-1.49785, 0.181956] (d=-0.554) | False |

- DiD metrics with CI excluding 0: 2/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=True (k=3 of n=4, direction=pos, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0860293 [-0.55338, 0.381322] (d=-0.13) | -2.08916 [-2.38409, -1.79423] (d=-5.01) |
| psd_peak_snr_db_mean | -1.41362 [-1.72394, -1.1033] (d=-3.22) | 0.0751262 [0.0439455, 0.106307] (d=1.7) |
| psd_peak_q_mean | 2.16868 [-0.216981, 4.55435] (d=0.643) | -4.87088 [-31.3013, 21.5595] (d=-0.13) |
| psd_peak_hz_iqr_hz | -2.26847 [-3.89961, -0.637325] (d=-0.983) | -4.84028 [-5.60119, -4.07937] (d=-4.5) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -2.00313 [-2.17555, -1.83071] (d=-8.21) | True |
| psd_peak_snr_db_mean | 1.48874 [1.20961, 1.76788] (d=3.77) | True |
| psd_peak_q_mean | -7.03956 [-35.8556, 21.7765] (d=-0.173) | False |
| psd_peak_hz_iqr_hz | -2.57182 [-3.44205, -1.70158] (d=-2.09) | True |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.750064 [0.508762, 0.991367] (d=2.2) | -1.12306 [-1.33305, -0.913065] (d=-3.78) |
| psd_peak_snr_db_mean | -0.932553 [-1.11094, -0.75417] (d=-3.7) | 0.294554 [0.174726, 0.414381] (d=1.74) |
| psd_peak_q_mean | 4.41737 [1.55031, 7.28442] (d=1.09) | -5.95604 [-7.39286, -4.51923] (d=-2.93) |
| psd_peak_hz_iqr_hz | 0.59436 [0.264653, 0.924066] (d=1.27) | -1.88194 [-1.91285, -1.85104] (d=-43.1) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -1.87312 [-1.90443, -1.84181] (d=-42.3) | True |
| psd_peak_snr_db_mean | 1.22711 [0.928896, 1.52532] (d=2.91) | True |
| psd_peak_q_mean | -10.3734 [-11.8037, -8.94317] (d=-5.13) | True |
| psd_peak_hz_iqr_hz | -2.4763 [-2.83692, -2.11569] (d=-4.86) | True |

- DiD metrics with CI excluding 0: 4/4

DECISION: PASS

