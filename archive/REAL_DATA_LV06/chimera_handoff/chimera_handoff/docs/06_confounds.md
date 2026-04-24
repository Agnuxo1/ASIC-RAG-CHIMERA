# 06 — Confounds

Confounds covered:

- batching (flush-period spectral peak)
- timing jitter (PSD broadening / peak degradation)

## Batching (flush-period)

Observed signature:

- `psd_peak_hz` concentrates near `1 / flush_period`.

Gate (C1, best-effort):

- When the burn condition is batching, the DiD report records `batching_expected_hz` and checks the median peak against it.

## Jitter (timestamp noise)

Observed signature:

- peak becomes broader, Q decreases, and error-to-target remains high.

Gate (C0, no false target lock):

- For confound conditions, the DiD report includes `C0_no_target_lock_pass` based on median `psd_peak_hz_error_hz_mean` exceeding preregistered tolerance.

