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

HF_DIR=${HF_DIR:-"${SLIME_ROOT}/init_ckpts/DeepDive-30B-SFT"}
OUT_DIR=${OUT_DIR:-"${SLIME_ROOT}/outputs/converted_ckpts/DeepDive-30B-SFT-torch_dist"}

mkdir -p "${OUT_DIR}"

PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
  --disable-bias-linear \
  --qk-layernorm \
  --group-query-attention \
  --num-attention-heads 32 \
  --num-query-groups 4 \
  --kv-channels 128 \
  --num-layers 48 \
  --hidden-size 2048 \
  --ffn-hidden-size 6144 \
  --normalization RMSNorm \
  --position-embedding-type rope \
  --norm-epsilon 1e-6 \
  --rotary-percent 1.0 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --vocab-size 152064 \
  --rotary-base 10000000 \
  --moe-ffn-hidden-size 768 \
  --moe-router-score-function softmax \
  --moe-token-dispatcher-type alltoall \
  --moe-router-topk 8 \
  --moe-layer-freq "([1]*48)" \
  --num-experts 128 \
  --moe-grouped-gemm \
  --moe-token-drop-policy probs \
  --moe-router-dtype fp32 \
  --moe-permute-fusion \
  --hf-checkpoint "${HF_DIR}" \
  --save "${OUT_DIR}"

echo "DONE: ${OUT_DIR}"
