#!/usr/bin/env bash
set -euo pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "${CARR_ROOT}/.env" ]]; then
  set -a
  source "${CARR_ROOT}/.env"
  set +a
fi
SLIME_ROOT="${SLIME_ROOT:-${CARR_ROOT}/slime}"

cd "${SLIME_ROOT}"

HF_DIR=${HF_DIR:-"${SLIME_ROOT}/init_ckpts/DeepDive-4B-SFT"}
OUT_DIR=${OUT_DIR:-"${SLIME_ROOT}/outputs/converted_ckpts/DeepDive-4B-SFT-torch_dist"}

mkdir -p "${OUT_DIR}"

PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
  --swiglu \
  --num-layers 36 \
  --hidden-size 2560 \
  --ffn-hidden-size 9728 \
  --num-attention-heads 32 \
  --group-query-attention \
  --num-query-groups 8 \
  --use-rotary-position-embeddings \
  --disable-bias-linear \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --rotary-base 5000000 \
  --vocab-size 152064 \
  --kv-channels 128 \
  --qk-layernorm \
  --hf-checkpoint "${HF_DIR}" \
  --save "${OUT_DIR}"

echo "DONE: ${OUT_DIR}"
