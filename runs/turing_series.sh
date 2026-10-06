#!/bin/bash
# Train a list of depths one after another on a single GPU, as one job array.
# Run on the login node from the repo root:
#   bash runs/turing_series.sh 1,3,6-9
set -euo pipefail
cd "$(dirname "$0")/.."

DEPTHS="${1:?usage: bash runs/turing_series.sh <depths, e.g. 1,3,6-9>}"

# Same precision everywhere so depth is the only thing that changes between runs.
export FP8=off
export WANDB_RUN_GROUP=depth-sweep

# %1 runs the array's tasks one at a time.
sbatch -J series --array="$DEPTHS%1" --output=%x-d%a-%A.out \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 runs/turing_base.sh
