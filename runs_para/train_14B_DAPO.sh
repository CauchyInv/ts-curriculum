#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export N_GPUS=${N_GPUS:-8}
export TP_SIZE=${TP_SIZE:-4}
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export ROOT=/hyk/algorithm_new/qinghua/yueyang/verl
export DATA_DIR=${DATA_DIR:-$ROOT/data}
export EXP_NAME=${EXP_NAME:-qwen3_14B_dapo_switchable}
export MODEL_PATH=${MODEL_PATH:-/hyk/algorithm_new/qinghua/yueyang/LUFFY/Qwen3_14B_Base}

# Keep v7/v8 toggles for compatibility with existing workflow.
export V7_FORMAT_MODE=${V7_FORMAT_MODE:-subproblem}   # subproblem | pn
export V7_NUM_PROBLEMS=${V7_NUM_PROBLEMS:-auto}       # auto | 1..4
export V7_REQUIRE_STRICT_EOS=${V7_REQUIRE_STRICT_EOS:-true}
export V7_PARSE_FAIL_POLICY=${V7_PARSE_FAIL_POLICY:-hard}  # hard | soft
export V7_T_MIX_MODE=${V7_T_MIX_MODE:-legacy_sub8}  # legacy_sub8 | balanced_2222 | mix62 | mix44 | balanced_11114 | mix44444
export V7_REWARD_MAP_MODE=${V7_REWARD_MAP_MODE:-legacy_k}  # legacy_k | absolute_difficulty
export V7_PROMPT_MODE=${V7_PROMPT_MODE:-explicit_t}  # explicit_t | unified
export V7_1_PROBLEM_MATCH=${V7_1_PROBLEM_MATCH:-v7}  # v7 | GRPO
export ADV_SHAPE_MODE=${ADV_SHAPE_MODE:-0}
export ADV_CLIP=${ADV_CLIP:--1}  # <=0 disabled; >0 clip token-level advantages to [-ADV_CLIP, ADV_CLIP]
export USE_ADAPTIVE=${USE_ADAPTIVE:-false}

# ---- DAPO baseline related knobs (main_ppo line) ----
export TS_VERSION=${TS_VERSION:-v9}  # v9 => pure GRPO path in your branch
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-true}

# Use DAPO reward manager and overlong penalty.
export REWARD_MANAGER=${REWARD_MANAGER:-dapo}  # dapo | naive | prime ...
export ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-true}
export OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-4096}
export OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
export OVERLONG_LOG=${OVERLONG_LOG:-false}

# Optional: force DAPO scorer via custom reward function.
# When true, we bypass default_compute_score routing and call math_dapo directly.
export USE_MATH_DAPO_SCORER=${USE_MATH_DAPO_SCORER:-false}

# DAPO filter_groups
export ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-true}
export FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}  # acc | score | seq_reward | seq_final_reward | ...
export MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}      # <=0 means no upper limit

if [ "$V7_FORMAT_MODE" = "subproblem" ]; then
  export USE_SUBPROBLEM_PROMPT=true
else
  export USE_SUBPROBLEM_PROMPT=false
fi

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
export LOG_PATH=./logs/$EXP_NAME.txt
export OMP_NUM_THREADS=1
export PYTHONPATH=/hyk/algorithm_new/qinghua/yueyang/verl:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export RAY_LOGGING_LEVEL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS=ignore

export MY_CUSTOM_TEMPLATE='{%- set _user_msg = messages | selectattr("role", "equalto", "user") | selectattr("content", "defined") | first -%}
{%- if _user_msg and "Please solve all 4 subproblems in order" in _user_msg["content"] -%}
    {%- for message in messages -%}
        {%- if message["role"] == "system" -%}
            {{- "<|im_start|>system\n" + message["content"] + "<|im_end|>\n" -}}
        {%- elif message["role"] == "user" -%}
            {{- "<|im_start|>user\n" + message["content"] + "<|im_end|>\n" -}}
        {%- elif message["role"] == "assistant" -%}
            {{- "<|im_start|>assistant\n" + message["content"] + "<|im_end|>\n" -}}
        {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
        {{- "<|im_start|>assistant\n**Subproblem 1**\n" -}}
    {%- endif -%}
{%- else -%}
    {%- for message in messages -%}
        {%- if message["role"] == "system" -%}
            {{- "<|im_start|>system\n" + message["content"] + "<|im_end|>\n" -}}
        {%- elif message["role"] == "user" -%}
            {{- "<|im_start|>user\n" + message["content"] + "<|im_end|>\n" -}}
        {%- elif message["role"] == "assistant" -%}
            {{- "<|im_start|>assistant\n" + message["content"] + "<|im_end|>\n" -}}
        {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
        {{- "<|im_start|>assistant\n" -}}
    {%- endif -%}
{%- endif -%}'

echo "[dapo] impl=main_ppo ts=$TS_VERSION adv_estimator=$ADV_ESTIMATOR reward_manager=$REWARD_MANAGER overlong_enable=$ENABLE_OVERLONG_BUFFER overlong_len=$OVERLONG_BUFFER_LEN overlong_penalty=$OVERLONG_PENALTY_FACTOR use_math_dapo_scorer=$USE_MATH_DAPO_SCORER"

COMMON_ARGS=(
  algorithm.adv_estimator="$ADV_ESTIMATOR"
  algorithm.kl_ctrl.kl_coef=0.000
  algorithm.norm_adv_by_std_in_grpo="$NORM_ADV_BY_STD_IN_GRPO"
  algorithm.use_kl_in_reward=False
  algorithm.filter_groups.enable="$ENABLE_FILTER_GROUPS"
  algorithm.filter_groups.metric="$FILTER_GROUPS_METRIC"
  algorithm.filter_groups.max_num_gen_batches="$MAX_NUM_GEN_BATCHES"
  data.train_files="${TRAIN_FILE:-$DATA_DIR/int/hard_1024_modified_system_prompt.parquet}"
  data.val_files="${VAL_FILE:-$DATA_DIR/valid_final.parquet}"
  data.train_batch_size=128
  data.val_batch_size=512
  data.max_prompt_length=1024
  data.max_response_length=8192
  data.filter_overlong_prompts=True
  data.truncation='error'
  +data.use_subproblem_prompt="$USE_SUBPROBLEM_PROMPT"
  +data.subproblems_jsonl_path=/hyk/algorithm_new/qinghua/yueyang/verl/data/int/hard_1024.jsonl
  +data.curri_method='teacher_student'
  +data.ts_version="'$TS_VERSION'"
  +data.v7_format_mode="'$V7_FORMAT_MODE'"
  +data.v7_num_problems="'$V7_NUM_PROBLEMS'"
  +data.v7_require_strict_eos="$V7_REQUIRE_STRICT_EOS"
  +data.v7_parse_fail_policy="'$V7_PARSE_FAIL_POLICY'"
  +data.v7_t_mix_mode="'$V7_T_MIX_MODE'"
  +data.v7_reward_map_mode="'$V7_REWARD_MAP_MODE'"
  +data.v7_prompt_mode="'$V7_PROMPT_MODE'"
  +data.v7_1_problem_match="'$V7_1_PROBLEM_MATCH'"
  +data.adv_shape_mode="$ADV_SHAPE_MODE"
  +data.adv_clip="$ADV_CLIP"
  +data.use_adaptive="$USE_ADAPTIVE"
  +data.teacher_model='none'
  +data.CREDIT_ASSIGNMENT_MODE=2
  ++data.all_problems_are_hard=true
  reward_model.reward_manager="$REWARD_MANAGER"
  +reward_model.reward_kwargs.max_resp_len=8192
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable="$ENABLE_OVERLONG_BUFFER"
  +reward_model.reward_kwargs.overlong_buffer_cfg.len="$OVERLONG_BUFFER_LEN"
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="$OVERLONG_PENALTY_FACTOR"
  +reward_model.reward_kwargs.overlong_buffer_cfg.log="$OVERLONG_LOG"
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.kl_loss_coef=0.00
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.000
  actor_rollout_ref.model.custom_chat_template="'$MY_CUSTOM_TEMPLATE'"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
  actor_rollout_ref.rollout.max_num_seqs=128
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.fsdp_config.param_offload=False
  trainer.critic_warmup=0
  trainer.logger="$TRAINER_LOGGER"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$EXP_NAME"
  trainer.val_before_train=False
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes=1
  trainer.validation_data_dir="$ROOT/validation_samples/$EXP_NAME"
  trainer.rollout_data_dir="$ROOT/rollout_samples/$EXP_NAME"
  +trainer.rollout_samples_dir="$ROOT/rollout_samples_own/$EXP_NAME"
  trainer.save_freq=50
  trainer.test_freq=10
  trainer.total_epochs=302
)

if [ "$USE_MATH_DAPO_SCORER" = "true" ]; then
  COMMON_ARGS+=(
    custom_reward_function.path=/hyk/algorithm_new/qinghua/yueyang/verl/exp_scripts/reward_math_dapo.py
    custom_reward_function.name=compute_score
  )
fi

python3 -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" "$@"
