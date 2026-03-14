#!/usr/bin/env bash
set -euo pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -f "${CARR_ROOT}/.env" ]]; then
  set -a
  source "${CARR_ROOT}/.env"
  set +a
fi

if [[ -z "${EVAL_DATA_PASSWORD:-}" ]]; then
  echo "EVAL_DATA_PASSWORD is required (set in .env or environment)." >&2
  exit 1
fi

python3 "${CARR_ROOT}/scripts/setup/file_crypto.py" encrypt-dir \
  --input-dir "${CARR_ROOT}/data/eval" \
  --output-dir "${CARR_ROOT}/data/eval" \
  --pattern "*.jsonl" \
  --password "${EVAL_DATA_PASSWORD}" \
  --suffix ".enc" \
  --remove-plain

echo "Encrypted eval files are ready in ${CARR_ROOT}/data/eval"
