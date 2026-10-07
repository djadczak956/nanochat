#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=djadczak@wpi.edu

# Fine-tune a trained base model into a chat model (SFT), then run the chat eval.
# Request as many GPUs as the base model trained on (checked below).
# Submit from the repo root:
#   DEPTH=24 sbatch -J sft-d24 runs/turing_sft.sh
#   DEPTH=4  sbatch -J sft-d4 --gres=gpu:L40S:1 --cpus-per-task=8 --mem=64G --time=02:00:00 runs/turing_sft.sh
# Several 1-GPU models one after another: bash runs/turing_sft_series.sh 1-3,5-9

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/projects/nanochat_data"
# In a job array, each task takes its depth from its array index.
DEPTH="${SLURM_ARRAY_TASK_ID:-${DEPTH:?set DEPTH, e.g. DEPTH=24}}"
WANDB_RUN="${WANDB_RUN:-sft-d$DEPTH}"

if curl -s --max-time 5 -o /dev/null https://api.wandb.ai; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

NGPU=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# SFT warm-starts from the base model's optimizer state, which is saved as one shard
# per GPU. A different GPU count would load the wrong shards.
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d$DEPTH"
PRETRAIN_GPUS=$(( $(ls "$CKPT_DIR" | sed -n 's/^optim_.*_rank\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1) + 1 ))
if (( NGPU != PRETRAIN_GPUS )); then
    echo "d$DEPTH was pretrained on $PRETRAIN_GPUS GPUs but this job has $NGPU; resubmit with --gres=gpu:L40S:$PRETRAIN_GPUS"
    exit 1
fi

echo "depth=$DEPTH node=$(hostname) gpus=${NGPU}x ${GPU_NAME} wandb=$WANDB_MODE run=$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NGPU" -m scripts.chat_sft -- \
    --model-tag="d$DEPTH" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NGPU" -m scripts.chat_eval -- -i sft -g "d$DEPTH"
