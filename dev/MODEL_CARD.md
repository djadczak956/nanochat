---
language:
- en
pipeline_tag: text-generation
tags:
- nanochat
- gpt
- depth-sweep
- interpretability
datasets:
- karpathy/climbmix-400b-shuffle
- HuggingFaceTB/smol-smoltalk
- cais/mmlu
- openai/gsm8k
---

# nanochat depth sweep

A family of small GPT language models that differ only in **depth** (number of transformer
blocks), trained with [nanochat](https://github.com/karpathy/nanochat) for a WPI Major
Qualifying Project on deep learning and dynamical systems. The project treats the residual
stream (the hidden state passed from block to block) as a trajectory and studies how it evolves
with depth, so every model is trained the same way and depth is the only thing that changes.

Each depth comes in two versions:

- **base**: pretrained on web text to predict the next token. It continues text; it does not follow instructions.
- **chat**: the base model after supervised fine-tuning (SFT), a short extra training run on example conversations that teaches it to answer as an assistant.

Training code and job scripts: [djadczak956/nanochat](https://github.com/djadczak956/nanochat) (`runs/turing_base.sh`, `runs/turing_sft.sh`).

## Models

| Depth | Width | Heads | Params (total) | Params (matrices + head) | Pretrain steps | Pretrain tokens | Val bpb ↓ | CORE ↑ | Chat steps | ChatCORE ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| d1  | 128  | 1  | 12.8M   | 4.4M   | 268   | 0.04B | 1.496 | −0.016 | 3,734 | 0.006 |
| d2  | 128  | 1  | 13.0M   | 4.6M   | 280   | 0.04B | 1.416 | −0.007 | 3,734 | 0.005 |
| d3  | 256  | 2  | 35.9M   | 10.7M  | 328   | 0.09B | 1.214 | 0.033  | 1,867 | 0.002 |
| d4  | 256  | 2  | 36.7M   | 11.5M  | 352   | 0.09B | 1.192 | 0.037  | 1,867 | 0.011 |
| d5  | 384  | 3  | 71.8M   | 21.4M  | 654   | 0.17B | 1.075 | 0.063  | 1,867 | 0.029 |
| d6  | 384  | 3  | 73.5M   | 23.2M  | 708   | 0.19B | 1.053 | 0.063  | 1,867 | 0.040 |
| d7  | 512  | 4  | 122.7M  | 38.8M  | 1,184 | 0.31B | 0.982 | 0.087  | 1,867 | 0.056 |
| d8  | 512  | 4  | 125.8M  | 41.9M  | 1,280 | 0.34B | 0.968 | 0.096  | 1,867 | 0.049 |
| d9  | 640  | 5  | 191.0M  | 65.2M  | 995   | 0.52B | 0.928 | 0.108  | 934   | 0.069 |
| d24 | 1536 | 12 | 1,384.1M | 729.8M | 5,568 | 5.84B | 0.718 | 0.267  | 467   | 0.204 |

Depths 10–23 are in progress.

- **Width** is depth × 64, rounded up to a multiple of 128 so every attention head is 128 wide. Deeper models are therefore also wider.
- **Params (total)** is dominated by embedding tables (one row per vocabulary token), which are lookups rather than computation. **Params (matrices + head)** counts only the weights that do matrix multiplies; nanochat sizes the training run from this number.
- **Val bpb** (bits per byte): how well the base model predicts held-out text, normalized by text length. Lower is better.
- **CORE**: average score over 22 benchmark tasks (HellaSwag, ARC, and others), centered so 0 is random guessing and 1 is perfect.
- **ChatCORE**: the same idea for chat models, over ARC-Easy, ARC-Challenge, MMLU, GSM8K and HumanEval.

## Architecture

nanochat's GPT, unchanged. Beyond a standard decoder-only transformer it has:

- rotary position embeddings and QK norm (queries and keys are normalized before attention)
- ReLU² MLPs (ReLU, then squared), 4× wide
- sliding-window attention in a repeating short-short-short-long pattern, with the last block always seeing the full 2,048-token context
- **value embeddings**: on alternating blocks, a per-token embedding is mixed into attention's values, giving those blocks direct access to token identity
- **x0 re-injection**: before each block, the state is rescaled and the initial embedding is added back in, `x ← λ_l·x + μ_l·x0`, with learned λ and μ
- **smear**: a small learned gate mixes each token's embedding with the previous token's
- **backout**: before the output head, a learned multiple of the middle block's output is subtracted
- logits soft-capped at ±15

The x0 re-injection matters for anyone studying hidden-state dynamics: a block's input is not simply the previous block's output.

## Training

**Pretraining.** ClimbMix web text, 2,048-token sequences, tokenizer: 32,768-token BPE trained on the same data. Each model sees 8 tokens per matrix parameter, slightly fewer than compute-optimal; batch size and learning rates are set automatically from model size. Muon optimizer for matrices and AdamW for embeddings, bf16 compute, no FP8. Trained on NVIDIA L40S GPUs on WPI's Turing cluster: one GPU for d1–d9, four for d24 (19 hours).

**SFT.** One pass over a mixture of SmolTalk (general conversations), MMLU (multiple choice, 3 copies) and GSM8K (math with a calculator tool, 4 copies), starting from the base model's weights and optimizer state.

Retraining with the same settings gives equally good models but not identical weights; GPU arithmetic is not deterministic.

## Usage

The repo mirrors nanochat's data directory, so a download can be used directly:

```bash
hf download damopierog/nanochat-depth-sweep --local-dir nanochat_data
export NANOCHAT_BASE_DIR=$PWD/nanochat_data
```

Then, from a nanochat checkout:

```python
import torch
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag="d9")   # or "sft" for the chat model
ids = tokenizer.encode("The capital of France is", prepend=tokenizer.get_bos_token_id())
print(tokenizer.decode(list(model.generate(ids, max_tokens=20, temperature=0.0))))
```

Chat with a chat model: `python -m scripts.chat_cli -i sft -g d24`.

## Files

```
base_checkpoints/d<N>/model_<step>.pt     base model weights
base_checkpoints/d<N>/meta_<step>.json    config and training metadata
chatsft_checkpoints/d<N>/...              chat models, same layout
tokenizer/                                tokenizer.pkl, token_bytes.pt
```

Only each model's final checkpoint is included, without optimizer state.

## Limitations

These are small research models. They state false facts confidently, repeat themselves, and have no safety training. d1–d4 chat models score at chance on ChatCORE, and many of their replies never end. Intended for studying model internals, not for use as assistants.
