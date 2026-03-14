#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

CARR_ROOT="${CARR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "${CARR_ROOT}/.env" ]]; then
  set -a
  source "${CARR_ROOT}/.env"
  set +a
fi
SLIME_ROOT="${SLIME_ROOT:-${CARR_ROOT}/slime}"

cd "${SLIME_ROOT}"

LOG_FILE="${SLIME_ROOT}/test_eval_4b.log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

export PYTHONBUFFERED=16

export TP_SIZE=4
export PP_SIZE=2
export CP_SIZE=4

ROLLOUT_TP_SIZE=8
ROLLOUT_MEM_UTILIZATION=0.8

MAX_CONTEXT_LEN=64000
MAX_GEN_LEN=64000
MAX_EVAL_CONTEXT_LEN=128000
MAX_EVAL_GEN_LEN=128000

EXP_TAG="eval-qwen3-4b-2507-rubric0.3-rop"
REF_MODEL_PATH="${REF_MODEL_PATH:-${SLIME_ROOT}/outputs/run-qwen3-4B-2507-deepdive-64k-rl-pif-rubric0.3-normal-rop/iter_0000419}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${SLIME_ROOT}/init_ckpts/DeepDive-4B-SFT}"
CKPT_DIR="${CKPT_DIR:-${SLIME_ROOT}/outputs/${EXP_TAG}}"
EVAL_DATA_ROOT=${CARR_ROOT}/data/eval_decrypted

MODEL_ARGS=(
  --max-position-embeddings ${MAX_CONTEXT_LEN}
  --seq-length ${MAX_CONTEXT_LEN}
  --swiglu
  --num-layers 36
  --hidden-size 2560
  --ffn-hidden-size 9728
  --num-attention-heads 32
  --group-query-attention
  --num-query-groups 8
  --use-rotary-position-embeddings
  --rotary-percent 1.0
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --rotary-base 5000000
  --vocab-size 152064
  --attention-softmax-in-fp32
  --attention-backend flash
  --accumulate-allreduce-grads-in-fp32
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --kv-channels 128
  --qk-layernorm
)

CKPT_ARGS=(
  --hf-checkpoint ${HF_MODEL_PATH}
  --ref-load ${REF_MODEL_PATH}
  --save-interval 1
  --save ${CKPT_DIR}
  --load ${CKPT_DIR}
  --ckpt-format torch_dist
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model ${HF_MODEL_PATH}
  --no-load-rng
  --no-load-optim
)

ROLLOUT_ARGS=(
  --custom-generate-function-path slime.rollout.rollout_with_tools_skip_illform.generate_with_tool
  --prompt-data ${EVAL_DATA_ROOT}/browsecomp.jsonl
  --source-key source
  --input-key input_messages
  --label-key label
  --tool-key tools
  --source-config-path configs/source_config.json
  --rollout-batch-size 8
  --n-samples-per-prompt 16
  --global-batch-size 128
  --num-rollout 3000
  --rollout-max-context-len ${MAX_CONTEXT_LEN}
  --rollout-max-response-len ${MAX_GEN_LEN}
  --rollout-temperature 1.0
  --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --sglang-server-concurrency 128
  --rollout-stop-token-ids 151643 151645
  --tool-max-retry 10
  --tool-timeout 300
  --only-eval
)

EVAL_ARGS=(
  --eval-interval 20
  --eval-prompt-data \
    browsecomp ${EVAL_DATA_ROOT}/browsecomp.jsonl \
    browsecomp-zh-3times ${EVAL_DATA_ROOT}/browsecomp_zh-3times.jsonl \
    xbench-deepsearch-3times ${EVAL_DATA_ROOT}/xbench-deepsearch-3times.jsonl \
    gaia-3times ${EVAL_DATA_ROOT}/gaia-3times.jsonl
  --n-samples-per-eval-prompt 1
  --eval-max-context-len ${MAX_EVAL_CONTEXT_LEN}
  --eval-max-response-len ${MAX_EVAL_GEN_LEN}
  --eval-temperature 1.0
  --eval-top-p 0.95
  --eval-top-k 40
  --eval-results-save-dir ${CKPT_DIR}/eval_results
)

DISTRIBUTED_ARGS=(
  --tensor-model-parallel-size ${TP_SIZE}
  --pipeline-model-parallel-size ${PP_SIZE}
  --context-parallel-size ${CP_SIZE}
  --sequence-parallel
)

PERF_ARGS=(
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
)

OPTIMIZER_ARGS=(
  --lr 2e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

export MASTER_ADDR=${MLP_WORKER_0_HOST}
export MASTER_PORT=${MLP_WORKER_0_PORT}
export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
  -- python3 train.py \
  --actor-num-nodes ${MLP_WORKER_NUM} \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus $(( ${MLP_WORKER_NUM} * 8 )) \
  --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE} \
  --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION} \
  --sglang-router-request-timeout-secs 360000 \
  --sglang-router-balance-abs-threshold 0 \
  --offload \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]}
