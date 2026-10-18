# Training the nanochat depth sweep on Turing

Instructions for anyone (or any Claude) running this fork on WPI's Turing cluster. Written
2026-10-12 by Damian Jadczak for the MQP on deep learning and dynamical systems.

## The project

We train nanochat base models at every depth 1–24 (depth = number of transformer blocks), with
identical settings so depth is the only variable, then study their hidden states (the residual
stream passed between blocks). Each depth also gets a chat version via SFT (supervised
fine-tuning on example conversations). Nothing in nanochat's model code is changed; the scripts
in `runs/turing_*.sh` only pick the depth and the cluster resources.

Pipeline per model: **pretraining** (`scripts/base_train.py`, almost all the compute) →
**SFT** (`scripts/chat_sft.py`, a few percent of that). The tokenizer is trained once and shared.

## Status (2026-10-18)

All base models d1–d24 are trained.

| Depths | Base | Chat (SFT) | GPUs used in pretraining |
|---|---|---|---|
| 1–9 | done | done | 1 |
| 10–12 | done | not yet | 1 |
| 13–14 | done (each trained twice by accident; the files on disk are from job 2351564) | not yet | 2 |
| 15–23 | done | not yet | 4 |
| 24 | done | done | 4 |

Final checkpoints (base d1–d24, chat d1–d9 and d24) are in the Hugging Face repo
`damopierog/nanochat-depth-sweep`. Its layout mirrors the data directory
below, so `hf download damopierog/nanochat-depth-sweep --local-dir <dir>` gives a usable
`NANOCHAT_BASE_DIR`.

## Setup

On the Turing login node (it has internet; compute nodes reached wandb and Hugging Face fine too):

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/djadczak956/nanochat.git
cd nanochat
uv sync --extra gpu
source .venv/bin/activate
wandb login                    # needs a recent wandb; the old pin rejected new-style API keys
```

Data lives outside the repo in `NANOCHAT_BASE_DIR`, default `~/projects/nanochat_data`; the job
scripts honor an exported value. It holds the pretraining shards, tokenizer, eval bundle, cached
SFT/eval datasets and all checkpoints.

**Use the same tokenizer as the existing models.** Every model's embeddings are tied to its token
IDs; a retrained tokenizer makes the existing checkpoints read text as garbage. Copy
`tokenizer/` from the Hugging Face repo (or from Damian's data directory) rather than running
`tok_train`.

Download the rest on the login node:

```bash
export NANOCHAT_BASE_DIR=$HOME/projects/nanochat_data
python -m nanochat.dataset -n 170          # ~17 GB of pretraining shards, enough through d24
python -c "from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle; from nanochat.common import download_file_with_lock; download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)"
```

Before the first SFT job, cache the SFT and chat-eval datasets:

```bash
python -c "
from tasks.smoltalk import SmolTalk; from tasks.mmlu import MMLU; from tasks.gsm8k import GSM8K
from tasks.arc import ARC; from tasks.humaneval import HumanEval
SmolTalk(split='train'); SmolTalk(split='test')
MMLU(subset='all', split='auxiliary_train'); MMLU(subset='all', split='test')
GSM8K(subset='main', split='train'); GSM8K(subset='main', split='test')
ARC(subset='ARC-Easy', split='test'); ARC(subset='ARC-Challenge', split='test'); HumanEval()
"
```

## The scripts

All are submitted from the repo root on the login node. Job notifications go to
`$USER@wpi.edu` through the series scripts; with plain `sbatch`, add `--mail-user=<you>@wpi.edu`.

| Script | Does |
|---|---|
| `runs/turing_base.sh` | Pretrain one depth (`DEPTH=N`), then run the base eval on it. Default resources are sized for d24 (4 GPUs, 24 h). |
| `runs/turing_series.sh <depths> [sbatch args]` | Pretrain several depths one at a time as a job array. Default 1 L40S, 4 h per depth; extra args override. |
| `runs/turing_sft.sh` | SFT one depth, then the chat eval. Stops immediately if the GPU count differs from pretraining's. |
| `runs/turing_sft_series.sh <depths>` | SFT several 1-GPU models one at a time. |
| `runs/turing_sweep.sh` | Old all-at-once plan for d1–d23; not used. |
| `dev/hf_upload.py <repo>` | Upload every model's final checkpoint plus the tokenizer to a private Hugging Face repo. |

Environment knobs read by `turing_base.sh`: `DEPTH`, `SAVE_EVERY` (default 500 steps),
`RESUME_STEP`, `WANDB_RUN` (default `d<depth>`), `FP8` (`auto|on|off`; the series scripts force
`off`).

Examples:

```bash
bash runs/turing_series.sh 10-12                                                        # small: 1 GPU each
bash runs/turing_series.sh 13-14 --gres=gpu:L40S:2 --cpus-per-task=16 --mem=128G --time=08:00:00
SAVE_EVERY=1000 bash runs/turing_series.sh 15-18 --gres=gpu:L40S:4 --cpus-per-task=32 --mem=256G --time=08:00:00
SAVE_EVERY=1000 bash runs/turing_series.sh 23 --gres=gpu:L40S:4 --cpus-per-task=32 --mem=256G --time=24:00:00
DEPTH=21 RESUME_STEP=3000 SAVE_EVERY=1000 sbatch -J d21-resume --gres=gpu:L40S:4 --cpus-per-task=32 --mem=256G runs/turing_base.sh
bash runs/turing_sft_series.sh 10-12
DEPTH=24 sbatch -J sft-d24 runs/turing_sft.sh                                           # 4-GPU model
```

Monitor: `squeue -u $USER`; logs are `<jobname>-d<depth>-<jobid>.out` in the repo root;
`grep -h "^depth=\|^CORE metric" series-d*.out` shows what each job ran on and its final score;
`sacct -u $USER -S <date> -X --format=JobID%18,JobName,State,Elapsed,AllocTRES%40` shows run times.

## Rules learned the hard way

- **`git pull` before every submit.** SLURM copies the script at submit time; an old copy once
  made a job evaluate the wrong model.
- **L40S only.** Jobs on the `rtx_pro_6000_b` (Blackwell) node crashed. The series scripts request
  `gpu:L40S`. Other GRES names on Turing: `A100`, `A30`, `H100`, `H200`, `L40S`.
- **Never submit the same depth twice.** Two runs of one depth write the same checkpoint folder;
  d13 and d14 were each trained twice this way. Check `squeue -u $USER` and `scancel` the old job
  before resubmitting a changed version.
- **FP8 stays off**, so every depth trains at the same precision (L40S cannot use it anyway).
- **SFT must use the same GPU count as that model's pretraining.** Optimizer state is saved as one
  file per GPU and SFT warm-starts from it. `turing_sft.sh` checks this.
- **Keep total use at about 4 GPUs at a time.** Batch the small models, run large ones alone.
- **Storage:** checkpoints include optimizer state (~60% of each file). Big models get
  `SAVE_EVERY=1000`. Copy only final `model_*.pt` + `meta_*.json` off the cluster.

## Cost (measured on L40S)

d24 took 19 h on 4 GPUs (76 GPU-hours). Cost grows roughly with depth to the fifth power:
d10–d16 ≈ 27 GPU-hours total, d17–d18 ≈ 30, d19–d22 ≈ 145, d23 ≈ 70. d1–d9 each train in
minutes to an hour on one GPU. SFT: ~16 min for d4 on 1 GPU, ~70 min for d24 on 4.

## Hidden states

`nanochat_viz/record.py` in the `inn_sequence` repo (branch `mqp26-27`) loads a checkpoint and records the residual
stream after each block (depth + 1 states) and after each attention / MLP half-step
(2·depth + 1). nanochat re-injects the embedding before every block (`x ← λ·x + μ·x0`), so a
block's input is not the previous block's output; the recorder stores block inputs separately
and measures updates from them.
