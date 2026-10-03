#!/bin/bash
#SBATCH --job-name=nanochat-d24
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# TIME_LIMIT_80 warns early enough to plan a resume before the job is killed.
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=djadczak@wpi.edu
# Uncomment and set once you know Turing's partitions/GPU types (`sinfo -o "%P %G %l %D"`):
# #SBATCH --partition=<partition>
# #SBATCH --constraint=<A100|H100>

# Pretrain a d24 nanochat base model on one Turing node.
# Submit from the repo root:   sbatch runs/turing_d24.sh
# Resume after a timeout:      RESUME_STEP=<step> sbatch runs/turing_d24.sh
# Name the wandb run:          WANDB_RUN=d24-try2 sbatch runs/turing_d24.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/projects/nanochat_data"
WANDB_RUN="${WANDB_RUN:-d24}"
RESUME_STEP="${RESUME_STEP:--1}"

# Log live if this node can reach wandb, otherwise write locally for `wandb sync` later.
if curl -s --max-time 5 -o /dev/null https://api.wandb.ai; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

NGPU=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# FP8 needs Hopper or newer.
FP8_FLAG=""
[[ "$GPU_NAME" =~ H100|H200|B200 ]] && FP8_FLAG="--fp8"

# Total batch size is fixed, so a smaller device batch only adds grad accumulation steps.
if (( GPU_MEM_MB >= 70000 )); then DBS=16; else DBS=8; fi

echo "node=$(hostname) gpus=${NGPU}x ${GPU_NAME} (${GPU_MEM_MB}MB) dbs=$DBS fp8=${FP8_FLAG:-off} wandb=$WANDB_MODE run=$WANDB_RUN resume=$RESUME_STEP"

torchrun --standalone --nproc_per_node="$NGPU" -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size="$DBS" \
    --save-every=500 \
    --resume-from-step="$RESUME_STEP" \
    --run="$WANDB_RUN" \
    $FP8_FLAG

torchrun --standalone --nproc_per_node="$NGPU" -m scripts.base_eval -- --device-batch-size="$DBS"

if [[ "$WANDB_MODE" == offline ]]; then
    echo "wandb ran offline. On the login node: cd $SLURM_SUBMIT_DIR && wandb sync wandb/offline-run-*"
fi
