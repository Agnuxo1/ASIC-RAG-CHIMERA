# 04 — Metrics

This package computes “Chronos metrics” on event timing streams and uses them in decision-grade DiD tests.

## Inputs

- `events.csv`: timestamped events (ns) plus optional metadata.
- `deltas.csv`: inter-event deltas `delta_s`.
- `chronos_metrics.csv`: sliding-window features over event windows.

## Windowed metrics (chronos)

Computed per window of `window_events` events:

- `cv`: coefficient of variation of `delta_s` in the window.
- `hist_entropy`: normalized histogram entropy of `log10(delta_s)` in the window.
- `hamming_norm`: mean normalized Hamming distance between consecutive `hash_hex` values (best-effort).
- `event_rate_hz`: `window_events / window_span_seconds`.

## PSD metrics

From the regular-binned count series in the window (bin width `psd_bin_dt_s`):

- `psd_peak_hz`: dominant frequency in `[f_min, f_max]`.
- `psd_peak_snr_db`: `10*log10(peak_power / median_band_power)`.
- `psd_peak_bandwidth_hz`: full-width at half-power around the peak (approx).
- `psd_peak_q`: `psd_peak_hz / psd_peak_bandwidth_hz` (0 if bandwidth undefined).
- `psd_peak_hz_error_hz`: `abs(psd_peak_hz - expected_heartbeat_hz)` when a target is known.

## PC‑CHRONOS v0.2 decision rule (PSD‑primary)

DiD uses (treated − surrogate) deltas and requires:

- Per-surrogate replication: at least `k` of the preregistered primary PSD metrics have CI excluding 0 and match preregistered polarity.
- Overall PASS: ≥1 IAAFT surrogate passes and ≥2 non‑IAAFT surrogates pass.

See `chimera_handoff/python/src/chimera_handoff/experiments/compute_did.py` for the exact logic.

