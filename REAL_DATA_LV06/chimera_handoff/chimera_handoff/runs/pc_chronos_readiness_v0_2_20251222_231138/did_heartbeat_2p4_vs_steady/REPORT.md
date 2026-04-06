# Difference-in-Differences (seed paired)

- Idle: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_steady_bits17`
- Burn: `/home/richard/Documents/heyting-dev-laptop-canonicalizewf/chimera_handoff/runs/pc_chronos_readiness_v0_2_20251222_231138/pc_chronos_v0_2_heartbeat_2p4_bits17`
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
| did | k-of-n same-direction |  | False |
| did | k-of-n CI excludes 0 | pc_pow_surrogate_phase_metric | False |
| v0.2 | overall |  | False |

## Surrogate: `pc_pow_surrogate_shuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=pos, passed=psd_peak_snr_db_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.282347 [-1.18681, 0.622112] (d=-0.221) | -0.497913 [-1.53647, 0.540648] (d=-0.339) |
| psd_peak_snr_db_mean | -0.914165 [-1.49249, -0.33584] (d=-1.12) | 0.469339 [0.199693, 0.738985] (d=1.23) |
| psd_peak_q_mean | -0.676101 [-0.935535, -0.416667] (d=-1.84) | 0.363095 [-5.91667, 6.64286] (d=0.0409) |
| psd_peak_hz_iqr_hz | 0.873808 [0.173433, 1.57418] (d=0.882) | -1.14061 [-1.38553, -0.895687] (d=-3.29) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.215565 [-2.15859, 1.72746] (d=-0.0784) | False |
| psd_peak_snr_db_mean | 1.3835 [0.535533, 2.23148] (d=1.15) | True |
| psd_peak_q_mean | 1.0392 [-4.98113, 7.05952] (d=0.122) | False |
| psd_peak_hz_iqr_hz | -2.01442 [-2.95971, -1.06912] (d=-1.51) | True |

- DiD metrics with CI excluding 0: 2/4

## Surrogate: `pc_pow_surrogate_blockshuffle_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.514604 [-0.169238, 1.19845] (d=0.532) | 1.3269 [0, 2.6538] (d=0.707) |
| psd_peak_snr_db_mean | -1.72351 [-3.12255, -0.324466] (d=-0.871) | -1.00706 [-2.01412, 0] (d=-0.707) |
| psd_peak_q_mean | 1.83798 [-1.23913, 4.91509] (d=0.422) | 3.42857 [0, 6.85714] (d=0.707) |
| psd_peak_hz_iqr_hz | 0.528741 [-0.440368, 1.49785] (d=0.386) | 2.43459 [0, 4.86918] (d=0.707) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.812294 [-1.19845, 2.82303] (d=0.286) | False |
| psd_peak_snr_db_mean | 0.716444 [-1.68966, 3.12255] (d=0.211) | False |
| psd_peak_q_mean | 1.59059 [-4.91509, 8.09627] (d=0.173) | False |
| psd_peak_hz_iqr_hz | 1.90585 [-1.49785, 5.30955] (d=0.396) | False |

- DiD metrics with CI excluding 0: 0/4

## Surrogate: `pc_pow_surrogate_phase_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=mixed, passed=psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | -0.0860293 [-0.55338, 0.381322] (d=-0.13) | 0.686957 [-0.631692, 2.00561] (d=0.368) |
| psd_peak_snr_db_mean | -1.41362 [-1.72394, -1.1033] (d=-3.22) | -0.130216 [-0.233391, -0.0270417] (d=-0.892) |
| psd_peak_q_mean | 2.16868 [-0.216981, 4.55435] (d=0.643) | 17.8929 [17.0714, 18.7143] (d=15.4) |
| psd_peak_hz_iqr_hz | -2.26847 [-3.89961, -0.637325] (d=-0.983) | 2.23656 [1.41128, 3.06184] (d=1.92) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.772987 [-1.01301, 2.55899] (d=0.306) | False |
| psd_peak_snr_db_mean | 1.2834 [1.07626, 1.49055] (d=4.38) | True |
| psd_peak_q_mean | 15.7242 [12.5171, 18.9313] (d=3.47) | True |
| psd_peak_hz_iqr_hz | 4.50503 [2.0486, 6.96145] (d=1.3) | True |

- DiD metrics with CI excluding 0: 3/4

## Surrogate: `pc_pow_surrogate_iaaft_metric`

- DiD prereg check (k-of-n, same direction): pass=False (k=3 of n=4, direction=pos, passed=psd_peak_snr_db_mean,psd_peak_q_mean)

### Within-condition diffs (thermal − surrogate)

| metric | idle mean_diff (ci95) | burn mean_diff (ci95) |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.750064 [0.508762, 0.991367] (d=2.2) | 0.819933 [0.80448, 0.835387] (d=37.5) |
| psd_peak_snr_db_mean | -0.932553 [-1.11094, -0.75417] (d=-3.7) | 0.545275 [0.378078, 0.712471] (d=2.31) |
| psd_peak_q_mean | 4.41737 [1.55031, 7.28442] (d=1.09) | 8.6631 [6.57619, 10.75] (d=2.94) |
| psd_peak_hz_iqr_hz | 0.59436 [0.264653, 0.924066] (d=1.27) | 1.19662 [-0.301757, 2.69499] (d=0.565) |

### DiD (Δthermal − Δsurrogate)

| metric | did mean (ci95) | ci_excludes_0 |
|---|---:|---:|
| psd_peak_hz_error_hz_mean | 0.0698689 [-0.186887, 0.326625] (d=0.192) | False |
| psd_peak_snr_db_mean | 1.47783 [1.13225, 1.82341] (d=3.02) | True |
| psd_peak_q_mean | 4.24573 [3.46558, 5.02588] (d=3.85) | True |
| psd_peak_hz_iqr_hz | 0.602259 [-0.56641, 1.77093] (d=0.364) | False |

- DiD metrics with CI excluding 0: 2/4

DECISION: FAIL

