#!/bin/bash
# Key params for launching with exp_scripts/train_4B_v7_switchable.sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
export N_GPUS=4
export TP_SIZE=4

export EXP_NAME=qwen3_4B_v8_mix44_absdiff_explicit_grpo_t1match_int1024_final_separate_usek_resp16384_gpu4567_live
export MODEL_PATH=/dev/shm/qinghua_models/Qwen3_4B_Base

export V7_FORMAT_MODE=pn
export V7_NUM_PROBLEMS=auto
export V7_REQUIRE_STRICT_EOS=false
export V7_PARSE_FAIL_POLICY=hard
export V7_T_MIX_MODE=mix44
export V7_REWARD_MAP_MODE=absolute_difficulty
export V7_PROMPT_MODE=explicit_t
export V7_1_PROBLEM_MATCH=GRPO
export ADV_CLIP=-1
export ACTOR_PPO_MICRO_BS=2
export ROLLOUT_LOGPROB_MICRO_BS=2
export REF_LOGPROB_MICRO_BS=2

# v8 separate + use_k
# launch cmd:
# bash /hyk/algorithm_new/qinghua/yueyang/verl/exp_scripts/train_4B_v7_switchable.sh \
#   data.ts_version=v8 \
#   data.train_files=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024_final.parquet \
#   data.max_response_length=16384 \
#   +data.v8_adv_group_mode=separate \
#   +data.use_k_as_subproblem_reward=true \
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
#   ++data.all_problems_are_hard=true \
#   ++data.subproblems_jsonl_path=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024.jsonl
