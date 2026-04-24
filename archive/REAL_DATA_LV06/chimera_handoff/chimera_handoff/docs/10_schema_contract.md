# 10 — Schema Contract

This contract is enforced by `chimera_handoff/python/src/chimera_handoff/schema.py` via `validate_run_root(path)`.

Schema version: `0.1`

## Run root layout (PC‑CHRONOS)

Required files at run root:

- `protocol.json` (must include `schema_version = "0.1"`)
- `preregistered_metrics.json`
- `manifest.json`
- `MANIFEST.sha256`
- `runs/<source>/seed_XX/...`

Required per-seed run directory:

- `config.json`
- `metrics.json`
- `events.csv`
- `deltas.csv`
- `chronos_metrics.csv`
- `chronos_metrics_meta.json`
- `chronos_summary.json`

## CSV headers (exact)

`events.csv`:

- `t_ns,nonce,hash_hex,difficulty_bits,attempts_since_prev,backend,notes_json`

`deltas.csv`:

- `t_ns,delta_s`

`chronos_metrics.csv`:

- `window_idx,window_t0_ns,window_t1_ns,cv,hist_entropy,hamming_norm,event_rate_hz,psd_peak_hz,psd_peak_power,psd_peak_snr_db,psd_peak_bandwidth_hz,psd_peak_q,psd_peak_hz_error_hz`

## Validation rules (selected)

- `events.t_ns` is strictly increasing.
- `delta_s` are positive.
- Headers match exactly (no extra/missing columns).

