#!/usr/bin/env bash
set -euo pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ ! -f "${CARR_ROOT}/.env" ]]; then
  cp "${CARR_ROOT}/.env.example" "${CARR_ROOT}/.env"
  echo "Created ${CARR_ROOT}/.env from .env.example"
else
  echo "${CARR_ROOT}/.env already exists"
fi

echo "Please edit ${CARR_ROOT}/.env before running training/evaluation scripts."
