#!/bin/bash
set -euo pipefail

source /export/miniconda3/etc/profile.d/conda.sh
conda activate curri

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export N_GPUS=${N_GPUS:-4}
export TP_SIZE=${TP_SIZE:-4}
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export ROOT=/hyk/algorithm_new/qinghua/yueyang/verl
export DATA_DIR=$ROOT/data
export EXP_NAME=${EXP_NAME:-qwen3_4B_v8_mix44_absdiff_explicit_grpo_t1match_int1024_final_separate_usek_validfinal_resp16384_micro222_gpu4567_live}
export MODEL_PATH=${MODEL_PATH:-/dev/shm/qinghua_models/Qwen3_4B_Base}

export V7_FORMAT_MODE=${V7_FORMAT_MODE:-pn}
export V7_NUM_PROBLEMS=${V7_NUM_PROBLEMS:-auto}
export V7_REQUIRE_STRICT_EOS=${V7_REQUIRE_STRICT_EOS:-false}
export V7_PARSE_FAIL_POLICY=${V7_PARSE_FAIL_POLICY:-hard}
export V7_T_MIX_MODE=${V7_T_MIX_MODE:-mix44}
export V7_REWARD_MAP_MODE=${V7_REWARD_MAP_MODE:-absolute_difficulty}
export V7_PROMPT_MODE=${V7_PROMPT_MODE:-explicit_t}
export V7_1_PROBLEM_MATCH=${V7_1_PROBLEM_MATCH:-GRPO}
export ADV_CLIP=${ADV_CLIP:--1}

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
export WANDB_API_KEY=${WANDB_API_KEY:-3e0863e2d8f819730b85529bd24b3ebbb96d0eb3}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-teacher_student_rl}
if [ "$WANDB_MODE" = "disabled" ]; then
  export TRAINER_LOGGER='["console"]'
else
  export TRAINER_LOGGER='["console","wandb"]'
fi

export RAY_memory_monitor_refresh_ms=0
export LOG_PATH=./logs/$EXP_NAME.log
export OMP_NUM_THREADS=1
export PYTHONPATH=/hyk/algorithm_new/qinghua/yueyang/verl:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com

cd /hyk/algorithm_new/qinghua/yueyang/verl

bash /hyk/algorithm_new/qinghua/yueyang/verl/exp_scripts/train_4B_v7_switchable.sh \
  data.ts_version=v8 \
  data.train_files=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024_final.parquet \
  data.val_files=/hyk/algorithm_new/qinghua/yueyang/verl/data/valid_final.parquet \
  data.max_response_length=16384 \
  +data.v8_adv_group_mode=separate \
  +data.use_k_as_subproblem_reward=true \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  ++data.all_problems_are_hard=true \
  ++data.subproblems_jsonl_path=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024.jsonl \
  "$@"
