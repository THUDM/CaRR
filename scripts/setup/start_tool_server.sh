#!/usr/bin/env bash
set -euo pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -f "${CARR_ROOT}/.env" ]]; then
  set -a
  source "${CARR_ROOT}/.env"
  set +a
fi

cd "${CARR_ROOT}/tool_server"

PROXY_ARGS=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
  PROXY_ARGS+=(--http_proxy "${HTTP_PROXY}")
fi

python3 launch_server.py \
  --serp_api_key "${SERP_API_KEY}" \
  --jina_api_key "${JINA_API_KEY}" \
  --port "${TOOL_SERVER_PORT:-7230}" \
  "${PROXY_ARGS[@]}"
