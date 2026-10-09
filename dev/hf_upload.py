"""
Upload the last checkpoint of every base and SFT model, plus the tokenizer, to a private
Hugging Face model repo. The repo mirrors $NANOCHAT_BASE_DIR, so

    hf download <repo> --local-dir <dir>

gives a directory that works as NANOCHAT_BASE_DIR. Optimizer state is skipped; it is only
needed to resume training. Rerunning uploads only what is new.

Run when no training job is in progress, or a half-trained model's latest checkpoint goes up:

    python -m dev.hf_upload <hf-username>/nanochat-depth-sweep
"""

import argparse
import os

from huggingface_hub import HfApi

from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir

parser = argparse.ArgumentParser(description="Upload final checkpoints to a private Hugging Face repo")
parser.add_argument("repo", help="e.g. <hf-username>/nanochat-depth-sweep")
args = parser.parse_args()

base_dir = get_base_dir()
patterns = ["tokenizer/*"]
for kind in ["base_checkpoints", "chatsft_checkpoints"]:
    root = os.path.join(base_dir, kind)
    if not os.path.isdir(root):
        continue
    for tag in sorted(os.listdir(root)):
        step = find_last_step(os.path.join(root, tag))
        patterns += [f"{kind}/{tag}/model_{step:06d}.pt", f"{kind}/{tag}/meta_{step:06d}.json"]
        print(f"{kind}/{tag}: step {step}")

HfApi().upload_large_folder(repo_id=args.repo, folder_path=base_dir, repo_type="model",
                            private=True, allow_patterns=patterns)
