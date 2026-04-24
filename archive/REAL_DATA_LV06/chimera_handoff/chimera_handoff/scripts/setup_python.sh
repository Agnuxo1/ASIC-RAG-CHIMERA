#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYDIR="${ROOT}/python"
VENV="${PYDIR}/.venv"

python3 -m venv "${VENV}"

# shellcheck disable=SC1090
source "${VENV}/bin/activate"

python -m pip install --upgrade "pip==25.2"
python -m pip install --require-hashes -r "${PYDIR}/requirements.lock"
python -m pip install -e "${PYDIR}"

python -c "import chimera_handoff; import numpy as _np; print('ok:', chimera_handoff.__file__)"

