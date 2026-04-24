#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Verifying MANIFEST.sha256 under ${ROOT}"
FOUND=0
while IFS= read -r -d '' mf; do
  FOUND=1
  d="$(dirname "${mf}")"
  b="$(basename "${mf}")"
  echo "--> ${mf}"
  (cd "${d}" && sha256sum -c "${b}")
done < <(find "${ROOT}" -name 'MANIFEST.sha256' -not -path '*/.venv/*' -not -path '*/.lake/*' -print0)

if [[ "${FOUND}" -eq 0 ]]; then
  echo "no MANIFEST.sha256 found under ${ROOT}" >&2
  exit 2
fi
