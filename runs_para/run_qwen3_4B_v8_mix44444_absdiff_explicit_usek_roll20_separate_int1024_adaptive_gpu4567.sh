#!/usr/bin/env bash
set -euo pipefail

cd /hyk/algorithm_new/qinghua/yueyang/verl

if [ -f /export/miniconda3/etc/profile.d/conda.sh ]; then
  source /export/miniconda3/etc/profile.d/conda.sh
elif [ -f /home/qinghua/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/qinghua/miniconda3/etc/profile.d/conda.sh
fi
conda activate curri

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export N_GPUS=${N_GPUS:-4}
export TP_SIZE=${TP_SIZE:-4}

TS=$(date +%Y%m%d_%H%M%S)
export EXP_NAME=${EXP_NAME:-qwen3_4B_v8_mix44444_absdiff_explicit_usek_roll20_separate_int1024_adaptive_gpu4567_${TS}}

export V7_FORMAT_MODE=${V7_FORMAT_MODE:-pn}
export V7_NUM_PROBLEMS=${V7_NUM_PROBLEMS:-auto}
export V7_REQUIRE_STRICT_EOS=${V7_REQUIRE_STRICT_EOS:-false}
export V7_PARSE_FAIL_POLICY=${V7_PARSE_FAIL_POLICY:-hard}
export V7_T_MIX_MODE=${V7_T_MIX_MODE:-mix44444}
export V7_REWARD_MAP_MODE=${V7_REWARD_MAP_MODE:-absolute_difficulty}
export V7_PROMPT_MODE=${V7_PROMPT_MODE:-explicit_t}
export V7_1_PROBLEM_MATCH=${V7_1_PROBLEM_MATCH:-GRPO}
export ADV_SHAPE_MODE=${ADV_SHAPE_MODE:-0}
export ADV_CLIP=${ADV_CLIP:--1}
export USE_ADAPTIVE=${USE_ADAPTIVE:-true}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}

export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-\
++data.ts_version=v8 \
++data.v8_adv_group_mode=separate \
++data.use_k_as_subproblem_reward=true \
++data.all_problems_are_hard=true \
++data.v8_mix44444_adaptive=true \
data.train_files=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024.parquet \
++data.subproblems_jsonl_path=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024.jsonl \
actor_rollout_ref.rollout.n=20 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.40}"

echo "[runs_para] config loaded for ${EXP_NAME}"
echo "[runs_para] start with:"
echo "bash exp_scripts/train_4B_v7_switchable.sh ${EXTRA_OVERRIDES}"
