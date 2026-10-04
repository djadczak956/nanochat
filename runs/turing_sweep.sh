#!/bin/bash
# Submit the depth 1-23 sweep as three job arrays, run one after another so the sweep
# never holds more than 4 GPUs at once. Run on the login node from the repo root:
#   bash runs/turing_sweep.sh
# Override depths (e.g. skip ones already trained): SMALL=1,3,4,6-12 bash runs/turing_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SMALL="${SMALL:-1-12}"
MEDIUM="${MEDIUM:-13-18}"
LARGE="${LARGE:-19-23}"

# Same precision everywhere so depth is the only thing that changes between runs.
export FP8=off
export WANDB_RUN_GROUP=depth-sweep

common=(--parsable --output=%x-d%a-%A.out)

# %N caps how many tasks of an array run at once.
a=$(SAVE_EVERY=50 sbatch "${common[@]}" -J sweep-small --array="$SMALL%4" \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 runs/turing_base.sh)
b=$(SAVE_EVERY=250 sbatch "${common[@]}" -J sweep-medium --array="$MEDIUM%2" \
    --dependency=afterany:$a \
    --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=12:00:00 runs/turing_base.sh)
c=$(SAVE_EVERY=500 sbatch "${common[@]}" -J sweep-large --array="$LARGE%1" \
    --dependency=afterany:$b \
    --gres=gpu:4 --cpus-per-task=32 --mem=256G --time=24:00:00 runs/turing_base.sh)

echo "small=$a medium=$b large=$c"
