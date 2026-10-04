#!/usr/bin/env bash
# Rebuild the ayanami-distill environment on a remote GPU machine
# (Colab / Kaggle / cloud) from requirements.lock.
#
# Strategy: install a GPU torch from the default PyPI index first, then
# install everything else pinned from requirements.lock, skipping the
# laptop-specific torch/nvidia/triton lines. This keeps one lock file
# while getting CUDA wheels on the remote side.
#
# Usage (on the remote machine, from the repo root):
#   bash scripts/setup_remote.sh
# Requirements on the remote side: python3 (>=3.10), git, network access.

set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYTHON_BIN:-python3}"
echo "[setup_remote] using interpreter: $($PYBIN --version 2>&1) ($PYBIN)"

if [ ! -d .venv ]; then
  echo "[setup_remote] creating .venv"
  "$PYBIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip

echo "[setup_remote] installing GPU torch from PyPI (default CUDA build)"
.venv/bin/python -m pip install --no-cache-dir torch

echo "[setup_remote] installing pinned stack (excluding laptop CPU-torch lines)"
grep -viE '^(torch|nvidia-|triton)[=<>]' requirements.lock > /tmp/ayanami_remote_reqs.txt || true
.venv/bin/python -m pip install --no-cache-dir -r /tmp/ayanami_remote_reqs.txt
echo "[setup_remote] installing bitsandbytes (CUDA-only, needed for 4-bit teacher)"
.venv/bin/python -m pip install --no-cache-dir bitsandbytes

echo "[setup_remote] verifying"
.venv/bin/python scripts/check_env.py
echo "[setup_remote] OK"
