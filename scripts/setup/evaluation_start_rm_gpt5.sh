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
  --port "${RM_EVAL_PORT:-6759}" \
  --model_name gpt-5-chat-2025-08-07 \
  --base_url "${OPENAI_BASE_URL}" \
  --api_key "${OPENAI_API_KEY}"
