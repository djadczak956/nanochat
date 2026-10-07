#!/bin/bash
# SFT a list of depths one after another on a single L40S, as one job array.
# Only for models pretrained on 1 GPU (turing_sft.sh checks this per task).
# Run on the login node from the repo root:
#   bash runs/turing_sft_series.sh 1-3,5-9
set -euo pipefail
cd "$(dirname "$0")/.."

DEPTHS="${1:?usage: bash runs/turing_sft_series.sh <depths, e.g. 1-3,5-9> [sbatch args...]}"
shift

# %1 runs the array's tasks one at a time.
sbatch -J sft-series --array="$DEPTHS%1" --output=%x-d%a-%A.out \
    --gres=gpu:L40S:1 --cpus-per-task=8 --mem=64G --time=04:00:00 "$@" runs/turing_sft.sh
