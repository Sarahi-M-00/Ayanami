#!/usr/bin/env python3
"""Fetch full-precision weights for the hoard (resumable via HF cache).

Usage:
    .venv/bin/python scripts/fetch_weights.py <repo_id> <dest_dir> [<revision>]

Downloads safetensors + tokenizer + config (skips .bin/.pt duplicates and
GGUFs, which are fetched separately). Re-running resumes partial blobs.
Prints total bytes at the end.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo = sys.argv[1]
dest = Path(sys.argv[2])
rev = sys.argv[3] if len(sys.argv) > 3 else None

dest.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo,
    revision=rev,
    local_dir=str(dest),
    allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.md", "*.txt"],
    max_workers=4,
)
total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
print(f"OK {repo} -> {path} ({total / 1e9:.1f} GB)")
