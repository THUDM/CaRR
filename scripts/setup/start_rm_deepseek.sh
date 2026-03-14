#!/usr/bin/env bash
set -euo pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -f "${CARR_ROOT}/.env" ]]; then
  set -a
  source "${CARR_ROOT}/.env"
  set +a
fi

cd "${CARR_ROOT}/deepsearch_rm_with_rubrics"

python3 launch_server.py \
  --port "${RM_TRAIN_PORT:-8888}" \
  --model_name deepseek-chat \
  --base_url "${DEEPSEEK_BASE_URL}" \
  --api_key "${DEEPSEEK_API_KEY}"
