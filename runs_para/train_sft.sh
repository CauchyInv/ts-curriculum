#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS=${N_GPUS:-4}
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export ROOT=/hyk/algorithm_new/qinghua/yueyang/verl
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

export MODEL_PATH=${MODEL_PATH:-/hyk/algorithm_new/qinghua/yueyang/LUFFY/Qwen3_4B_Base}
export TRAIN_FILE=${TRAIN_FILE:-$ROOT/data/int/hard_1024_sft_ref_nonempty_with_boxed.parquet}
export VAL_FILE=${VAL_FILE:-$TRAIN_FILE}

export EXP_NAME=${EXP_NAME:-qwen3_4B_sft_int_ref_nonempty_with_boxed}
export SFT_EPOCHS=${SFT_EPOCHS:-10000}
export SFT_LR=${SFT_LR:-1e-6}
export SFT_MAX_LEN=${SFT_MAX_LEN:-8192}
export SFT_TRAIN_BS=${SFT_TRAIN_BS:-32}
export SFT_MICRO_BS=${SFT_MICRO_BS:-1}
export SFT_TEST_FREQ=${SFT_TEST_FREQ:-20}
export SFT_SAVE_FREQ=${SFT_SAVE_FREQ:-30}
export SFT_SAMPLES_ENABLE=${SFT_SAMPLES_ENABLE:-true}
export SFT_SAMPLES_MAX=${SFT_SAMPLES_MAX:-128}
export SFT_SAMPLES_ONLY_FIRST_EPOCH=${SFT_SAMPLES_ONLY_FIRST_EPOCH:-true}
export SFT_SAMPLES_DIR=${SFT_SAMPLES_DIR:-$ROOT/rollout_samples_own/$EXP_NAME}
export SFT_IGNORE_INPUT_IDS_MISMATCH=${SFT_IGNORE_INPUT_IDS_MISMATCH:-true}

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
export WANDB_API_KEY=${WANDB_API_KEY:-3e0863e2d8f819730b85529bd24b3ebbb96d0eb3}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-teacher_student_rl}
if [ "$WANDB_MODE" = "disabled" ]; then
    export TRAINER_LOGGER='["console"]'
else
    export TRAINER_LOGGER='["console","wandb"]'
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] TRAIN_FILE not found: $TRAIN_FILE"
    exit 1
fi

echo "[SFT] model=$MODEL_PATH"
echo "[SFT] train=$TRAIN_FILE"
echo "[SFT] val=$VAL_FILE"
echo "[SFT] exp=$EXP_NAME"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.multiturn.enable=true \
    +data.messages_key=messages \
    +data.ignore_input_ids_mismatch=$SFT_IGNORE_INPUT_IDS_MISMATCH \
    data.max_length=$SFT_MAX_LEN \
    data.truncation=right \
    data.train_batch_size=$SFT_TRAIN_BS \
    data.micro_batch_size_per_gpu=$SFT_MICRO_BS \
    model.partial_pretrain=$MODEL_PATH \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    use_remove_padding=True \
    optim.lr=$SFT_LR \
    trainer.total_epochs=$SFT_EPOCHS \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$ROOT/checkpoints/$WANDB_PROJECT/$EXP_NAME \
    trainer.save_freq=$SFT_SAVE_FREQ \
    trainer.test_freq=$SFT_TEST_FREQ \
    +trainer.sft_samples_enable=$SFT_SAMPLES_ENABLE \
    +trainer.sft_samples_max=$SFT_SAMPLES_MAX \
    +trainer.sft_samples_only_first_epoch=$SFT_SAMPLES_ONLY_FIRST_EPOCH \
    +trainer.sft_samples_dir=$SFT_SAMPLES_DIR \
    "$@"
