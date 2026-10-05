#!/usr/bin/env python3
"""Trainer CLI (Phase 5).

Usage:
  PYTHONPATH=src .venv/bin/python -m ayanami_distill.train \
      --config configs/distill/text_sft.yaml [--run-dir runs/<run_id>]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .trainer import ROOT, train_logit_kd, train_text_sft


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ayanami distillation trainer")
    ap.add_argument("--config", required=True, help="distill config YAML")
    ap.add_argument("--run-dir", default=None, help="output dir (default: runs/<ts>)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mode = cfg.get("mode")
    if mode == "text_sft":
        train_fn = train_text_sft
    elif mode == "logit_kd":
        train_fn = train_logit_kd
    else:
        print(f"mode {mode!r}: unknown training mode", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir or
                   ("runs/" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    + "_" + cfg["mode"]))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary = train_fn(cfg, run_dir)
    print(f"done: {run_dir} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
