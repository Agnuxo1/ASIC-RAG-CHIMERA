#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${ROOT}/python/.venv" ]]; then
  "${ROOT}/scripts/setup_python.sh"
fi

# shellcheck disable=SC1090
source "${ROOT}/python/.venv/bin/activate"

if ! ls /sys/class/thermal/thermal_zone*/temp >/dev/null 2>&1; then
  echo "thermal sensors not found under /sys; skipping thermal stream run"
  exit 0
fi

TAG="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/runs/thermal_stream_v1_2_${TAG}"

python -m chimera_handoff.experiments.sweep_thermal_stream \
  --out "${OUT}" \
  --seeds "0-1" \
  --thermal-duration 60.0 \
  --thermal-dt 0.05 \
  --thermal-calib-seconds 10.0 \
  --thermal-intervention none

echo "out_root: ${OUT}"

