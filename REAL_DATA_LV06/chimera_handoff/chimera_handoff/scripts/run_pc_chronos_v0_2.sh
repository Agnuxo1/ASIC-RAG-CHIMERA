#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${ROOT}/python/.venv" ]]; then
  "${ROOT}/scripts/setup_python.sh"
fi

# shellcheck disable=SC1090
source "${ROOT}/python/.venv/bin/activate"

TAG="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${ROOT}/runs/pc_chronos_readiness_v0_2_${TAG}"
mkdir -p "${OUT_BASE}"

SEEDS="${CHIMERA_SEEDS:-10-19}"
DURATION="${CHIMERA_DURATION:-120.0}"
BITS="${CHIMERA_BITS:-17}"
WINDOW_EVENTS="${CHIMERA_WINDOW_EVENTS:-64}"

if [[ "${CHIMERA_FAST:-0}" == "1" ]]; then
  SEEDS="0-1"
  DURATION="20.0"
  WINDOW_EVENTS="16"
  ALLOW_REUSE="--allow-seed-reuse"
else
  ALLOW_REUSE=""
fi

STEADY="${OUT_BASE}/pc_chronos_v0_2_steady_bits${BITS}"
HB_2P4="${OUT_BASE}/pc_chronos_v0_2_heartbeat_2p4_bits${BITS}"
HB_3P7="${OUT_BASE}/pc_chronos_v0_2_heartbeat_3p7_bits${BITS}"
BATCH="${OUT_BASE}/pc_chronos_v0_2_batching_200ms_bits${BITS}"
JITTER="${OUT_BASE}/pc_chronos_v0_2_jitter_2ms_bits${BITS}"

echo "==> Sweep: steady"
chimera-pc-chronos-sweep ${ALLOW_REUSE} --protocol-version pc_chronos_v0_2 --condition steady --seeds "${SEEDS}" --duration "${DURATION}" --difficulty-bits "${BITS}" --metrics-window-events "${WINDOW_EVENTS}" --out "${STEADY}"
chimera-summarize --in "${STEADY}" --out "${STEADY}/summary_prng"

echo "==> Sweep: heartbeat 2.4"
chimera-pc-chronos-sweep ${ALLOW_REUSE} --protocol-version pc_chronos_v0_2 --condition heartbeat --heartbeat-hz 2.4 --seeds "${SEEDS}" --duration "${DURATION}" --difficulty-bits "${BITS}" --metrics-window-events "${WINDOW_EVENTS}" --out "${HB_2P4}"
chimera-summarize --in "${HB_2P4}" --out "${HB_2P4}/summary_prng"

echo "==> Sweep: heartbeat 3.7"
chimera-pc-chronos-sweep ${ALLOW_REUSE} --protocol-version pc_chronos_v0_2 --condition heartbeat --heartbeat-hz 3.7 --seeds "${SEEDS}" --duration "${DURATION}" --difficulty-bits "${BITS}" --metrics-window-events "${WINDOW_EVENTS}" --out "${HB_3P7}"
chimera-summarize --in "${HB_3P7}" --out "${HB_3P7}/summary_prng"

echo "==> Sweep: batching (200ms)"
chimera-pc-chronos-sweep ${ALLOW_REUSE} --protocol-version pc_chronos_v0_2 --condition batching --batch-flush-ms 200 --seeds "${SEEDS}" --duration "${DURATION}" --difficulty-bits "${BITS}" --metrics-window-events "${WINDOW_EVENTS}" --out "${BATCH}"
chimera-summarize --in "${BATCH}" --out "${BATCH}/summary_prng"

echo "==> Sweep: jitter (±2ms)"
chimera-pc-chronos-sweep ${ALLOW_REUSE} --protocol-version pc_chronos_v0_2 --condition jitter --jitter-ms 2 --seeds "${SEEDS}" --duration "${DURATION}" --difficulty-bits "${BITS}" --metrics-window-events "${WINDOW_EVENTS}" --out "${JITTER}"
chimera-summarize --in "${JITTER}" --out "${JITTER}/summary_prng"

METRICS="psd_peak_hz_error_hz_mean,psd_peak_snr_db_mean,psd_peak_q_mean,psd_peak_hz_iqr_hz"
SURR="pc_pow_surrogate_shuffle_metric,pc_pow_surrogate_blockshuffle_metric,pc_pow_surrogate_phase_metric,pc_pow_surrogate_iaaft_metric"
TREAT="pc_pow_share_events_metric"

echo "==> DiD: heartbeat_2p4 vs steady (v0.2 PSD-primary)"
chimera-did --idle "${STEADY}" --burn "${HB_2P4}" --out "${OUT_BASE}/did_heartbeat_2p4_vs_steady" --treatment-source "${TREAT}" --surrogates "${SURR}" --metrics "${METRICS}" --decision-rule v0_2_psd_primary

echo "==> DiD: heartbeat_3p7 vs steady (v0.2 PSD-primary)"
chimera-did --idle "${STEADY}" --burn "${HB_3P7}" --out "${OUT_BASE}/did_heartbeat_3p7_vs_steady" --treatment-source "${TREAT}" --surrogates "${SURR}" --metrics "${METRICS}" --decision-rule v0_2_psd_primary

echo "==> DiD: batching vs steady (confound gate)"
chimera-did --idle "${STEADY}" --burn "${BATCH}" --out "${OUT_BASE}/did_batching_vs_steady" --treatment-source "${TREAT}" --surrogates "${SURR}" --metrics "${METRICS}" --decision-rule v0_2_psd_primary

echo "==> DiD: jitter vs steady (confound gate)"
chimera-did --idle "${STEADY}" --burn "${JITTER}" --out "${OUT_BASE}/did_jitter_vs_steady" --treatment-source "${TREAT}" --surrogates "${SURR}" --metrics "${METRICS}" --decision-rule v0_2_psd_primary

cat > "${OUT_BASE}/FINAL_REPORT.md" <<EOF
# PC-CHRONOS Readiness v0.2 (handoff) — Final Report

## Run roots (seeds ${SEEDS}, duration ${DURATION}s)

- Steady: \`${STEADY}\`
- Heartbeat 2.4 Hz: \`${HB_2P4}\`
- Heartbeat 3.7 Hz: \`${HB_3P7}\`
- Batching confound (200 ms flush): \`${BATCH}\`
- Jitter confound (±2 ms): \`${JITTER}\`

## Decision artifacts (v0.2 PSD-primary)

All DiD roots below use:
- treatment source: \`${TREAT}\`
- surrogates: \`${SURR}\`
- metrics: \`${METRICS}\`
- decision rule: \`v0_2_psd_primary\`

- Heartbeat 2.4 vs steady: \`${OUT_BASE}/did_heartbeat_2p4_vs_steady/REPORT.md\`
- Heartbeat 3.7 vs steady: \`${OUT_BASE}/did_heartbeat_3p7_vs_steady/REPORT.md\`
- Batching vs steady: \`${OUT_BASE}/did_batching_vs_steady/REPORT.md\`
- Jitter vs steady: \`${OUT_BASE}/did_jitter_vs_steady/REPORT.md\`
EOF

echo "wrote: ${OUT_BASE}/FINAL_REPORT.md"
echo "out_root: ${OUT_BASE}"
