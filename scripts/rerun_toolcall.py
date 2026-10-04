#!/usr/bin/env python3
"""Stage 10.1 — Re-run the FIXED tool-call suite only (10 gens, <10 min).

Uses harness.suite_toolcall (tools-in-context + strict parse_ok).
Writes runs/eval0/toolcall_v2.json.
"""

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="runs/eval0/toolcall_v2.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ayanami_distill.eval.harness import suite_toolcall

    torch.manual_seed(7)
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(str(ROOT / "student_base"), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "student_base"), dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    trajs = [json.loads(line) for line in
             open(ROOT / "tests/fixtures/trajectories.jsonl", encoding="utf-8")]
    trajs = trajs[args.start:args.start + args.limit]
    result = suite_toolcall(model, tok, trajs)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
