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

LOG_FILE="${SLIME_ROOT}/test_30b.log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

export PYTHONBUFFERED=16
export PYTHONUNBUFFERED=1

export TP_SIZE=4
export PP_SIZE=2
export CP_SIZE=4
export EP_SIZE=16
export ETP_SIZE=1

ROLLOUT_TP_SIZE=8
ROLLOUT_MEM_UTILIZATION=0.8

MAX_CONTEXT_LEN=64000
MAX_GEN_LEN=64000
MAX_EVAL_CONTEXT_LEN=64000
MAX_EVAL_GEN_LEN=64000

EXP_TAG="run-qwen3-30B-A3B-2507-search_deepdive-64k-rubric0.3-normal-rop"
HF_MODEL_PATH="${HF_MODEL_PATH:-${SLIME_ROOT}/init_ckpts/DeepDive-30B-SFT}"
REF_MODEL_TORCH_DIST="${REF_MODEL_TORCH_DIST:-${SLIME_ROOT}/outputs/converted_ckpts/DeepDive-30B-SFT-torch_dist}"
CKPT_DIR="${CKPT_DIR:-${SLIME_ROOT}/outputs/${EXP_TAG}}"
EVAL_DATA_ROOT=${CARR_ROOT}/data/eval_decrypted
PROMPT_DATA="${PROMPT_DATA:-${SLIME_ROOT}/outputs/data/deepdive-rl-2k-browser-oss-rubric.jsonl}"

MODEL_ARGS=(
  --max-position-embeddings ${MAX_CONTEXT_LEN}
  --seq-length ${MAX_CONTEXT_LEN}
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 32
  --num-query-groups 4
  --kv-channels 128
  --num-layers 48
  --hidden-size 2048
  --ffn-hidden-size 6144
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --normalization RMSNorm
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 1.0
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 152064
  --rotary-base 10000000
  --attention-softmax-in-fp32
  --attention-backend flash
  --moe-ffn-hidden-size 768
  --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 8
  --moe-layer-freq 1
  --num-experts 128
  --moe-grouped-gemm
  --moe-token-drop-policy probs
  --moe-router-dtype fp32
  --moe-permute-fusion
)

CKPT_ARGS=(
  --hf-checkpoint ${HF_MODEL_PATH}
  --ref-load ${REF_MODEL_TORCH_DIST}
  --save-interval 1
  --save ${CKPT_DIR}
  --load ${CKPT_DIR}
  --keep-optim-recent 1
  --ckpt-format torch_dist
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model ${HF_MODEL_PATH}
  --no-load-rng
)

ROLLOUT_ARGS=(
  --custom-generate-function-path slime.rollout.rollout_with_tools_skip_illform.generate_with_tool
  --prompt-data ${PROMPT_DATA}
  --source-key source
  --input-key input_messages
  --label-key label
  --tool-key tools
  --source-config-path configs/source_config.json
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --global-batch-size 128
  --num-rollout 3000
  --rollout-max-context-len ${MAX_CONTEXT_LEN}
  --rollout-max-response-len ${MAX_GEN_LEN}
  --rollout-shuffle
  --rollout-temperature 1.0
  --rollout-top-p 0.95
  --rollout-top-k 40
  --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --sglang-server-concurrency 128
  --rollout-stop-token-ids 151643 151645
  --tool-max-retry 5
  --tool-timeout 300
  --stop-once-illform
  --rubric-reward-ratio 0.3
  --normalize-rubric-reward
  --rubric-only-positive
)

EVAL_ARGS=(
  --skip-eval-before-train
  --eval-interval 20
  --eval-prompt-data \
    browsecomp266 ${EVAL_DATA_ROOT}/browsecomp266.jsonl \
    xbench-deepsearch ${EVAL_DATA_ROOT}/xbench-deepsearch.jsonl \
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
  --expert-model-parallel-size ${EP_SIZE}
  --expert-tensor-parallel-size ${ETP_SIZE}
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
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --calculate-per-token-loss
  --use-tis
  --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function
  --tis-clip-low 0.5
  --tis-clip 2.0
)

OPTIMIZER_ARGS=(
  --lr 2e-6
  --lr-warmup-iters 0
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --override-opt_param-scheduler
)

WANDB_ARGS=(
  --use-wandb
  --wandb-key ${WANDB_API_KEY}
  --wandb-project dr-rl-qwen3
  --wandb-group ${EXP_TAG}
  --disable-wandb-random-suffix
  --wandb-always-use-train-step
)

export MASTER_ADDR=${MLP_WORKER_0_HOST}
export MASTER_PORT=${MLP_WORKER_0_PORT}
export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}

mkdir -p "${CKPT_DIR}"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

WORKER_PIDS=()
for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
   continue
  fi
  ssh root@"${WORKER_IP}" \
   "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  WORKER_PIDS+=("$!")
done
for WORKER_PID in "${WORKER_PIDS[@]}"; do
  wait "${WORKER_PID}"
done

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
    "env_vars": {
      "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
      "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
      "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
      "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
      "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
      "MASTER_PORT": "${MLP_WORKER_0_PORT}",
      "PYTHONPATH": "/root/Megatron-LM/",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1"
    }
  }' \
  -- python3 train.py \
  --actor-num-nodes ${MLP_WORKER_NUM} \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus $(( ${MLP_WORKER_NUM} * 8 )) \
  --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE} \
  --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION} \
  --sglang-ep-size 8 \
  --sglang-router-request-timeout-secs 36000 \
  --sglang-router-balance-abs-threshold 0 \
  --offload \
  --colocate \
  --no-check-for-nan-in-loss-and-grad \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]}
