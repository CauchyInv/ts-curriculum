#!/bin/bash
set -euo pipefail

# Usage:
#   bash eval_models/run_eval_passk.sh /ckpt/path1 [/ckpt/path2 ...]
# Or:
#   CKPTS="/ckpt/a /ckpt/b" bash eval_models/run_eval_passk.sh

export ROOT=${ROOT:-/hyk/algorithm_new/qinghua/yueyang/verl}
export PARQUET=${PARQUET:-$ROOT/data/valid.parquet}
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/eval_models/results}

export N_SAMPLES=${N_SAMPLES:-64}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
export VAL_TOP_K=${VAL_TOP_K:--1}
export VAL_TOP_P=${VAL_TOP_P:-1.0}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export PASSK_MODE=${PASSK_MODE:-combinatorial}  # empirical | combinatorial | verl_bootstrap | both | both_all
export SAVE_ROLLOUT=${SAVE_ROLLOUT:-true}       # true | false
export ROLLOUT_DIR=${ROLLOUT_DIR:-$ROOT/eval_models/eval_rollout_samples}

export TP_SIZE=${TP_SIZE:-1}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export DTYPE=${DTYPE:-auto}
export BATCH_SIZE=${BATCH_SIZE:-32}

if [ "$#" -gt 0 ]; then
  CKPTS="$*"
else
  CKPTS=${CKPTS:-}
fi

if [ -z "${CKPTS}" ]; then
  echo "[error] No checkpoints provided."
  echo "Provide ckpts either by args or CKPTS env."
  echo "Example:"
  echo "  bash $ROOT/eval_models/run_eval_passk.sh /path/ckpt_a /path/ckpt_b"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[info] ROOT=${ROOT}"
echo "[info] PARQUET=${PARQUET}"
echo "[info] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[info] CKPTS=${CKPTS}"
echo "[info] n=${N_SAMPLES}, temp=${VAL_TEMPERATURE}, top_k=${VAL_TOP_K}, top_p=${VAL_TOP_P}, max_tokens=${MAX_TOKENS}"
echo "[info] passk_mode=${PASSK_MODE}"
echo "[info] save_rollout=${SAVE_ROLLOUT}, rollout_dir=${ROLLOUT_DIR}"
echo "[info] tp=${TP_SIZE}, gpu_mem_util=${GPU_MEM_UTIL}, max_model_len=${MAX_MODEL_LEN}, dtype=${DTYPE}, batch_size=${BATCH_SIZE}"

SAVE_ROLLOUT_ARG="--save_rollout"
if [ "${SAVE_ROLLOUT}" = "false" ] || [ "${SAVE_ROLLOUT}" = "0" ]; then
  SAVE_ROLLOUT_ARG="--no-save_rollout"
fi

python "$ROOT/eval_models/eval_passk.py" \
  --ckpts ${CKPTS} \
  --parquet "${PARQUET}" \
  --output_dir "${OUTPUT_DIR}" \
  --n "${N_SAMPLES}" \
  --temperature "${VAL_TEMPERATURE}" \
  --top_k "${VAL_TOP_K}" \
  --top_p "${VAL_TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --passk_mode "${PASSK_MODE}" \
  ${SAVE_ROLLOUT_ARG} \
  --rollout_dir "${ROLLOUT_DIR}" \
  --tensor_parallel_size "${TP_SIZE}" \
  --gpu_memory_utilization "${GPU_MEM_UTIL}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}"

echo "[done] pass@k evaluation finished."
