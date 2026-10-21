#!/usr/bin/env python3
"""Stage 10.6 — Dev-only evaluation of a merged model (A@800 vs B@800).

Uses ONLY the non-sealed dev split (`scripts/dev_cases.py`), so the sealed
test roster stays untouched for the one-time final run. Suites and decoding
match BASELINE0_v2: persona 120 new tokens, identity 60, others 80, greedy,
thinking off, seed 7. PPL is the 60-line / 48-token strided FineWeb slice.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/eval_dev.py \
      --model exports/merged_B800 --out runs/eval_B800_dev/eval.json \
      --device cuda
"""

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MAX_NEW = {"persona": 120, "identity": 60}
DEFAULT_MAX_NEW = 80
DEV_DIR = Path("/tmp/ayanami_dev")
SUITES = ("persona", "obedience", "identity", "injection",
          "devops", "security", "sandbox")


def materialize_dev() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/dev_cases.py")],
                   check=True, cwd=str(ROOT))


def summarize(results: list[dict]) -> dict:
    buckets: dict[str, list[str]] = {"overall:all": []}
    for r in results:
        buckets.setdefault(f"cat:{r['category']}", []).append(r["verdict"])
        buckets.setdefault(f"lang:{r['lang']}", []).append(r["verdict"])
        buckets["overall:all"].append(r["verdict"])
    out: dict = {"n": len(results), "by_category": {}, "by_lang": {},
                 "overall": {}}
    for name, verdicts in buckets.items():
        kind, label = name.split(":", 1)
        entry = {v: verdicts.count(v) for v in ("pass", "fail", "review")}
        entry["pass_rate_auto"] = round(
            entry["pass"] / max(1, entry["pass"] + entry["fail"]), 4)
        if kind == "overall":
            out["overall"] = entry
        elif kind == "cat":
            out["by_category"][label] = entry
        else:
            out["by_lang"][label] = entry
    return out


def gen_suite(model, tok, cases: list[dict], max_new: int, device: str,
              sys_prompt: str | None) -> list[dict]:
    import torch
    from ayanami_distill.eval.persona_eval import apply_check

    rows = []
    for i, case in enumerate(cases):
        messages = list(case["messages"])
        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}] + messages
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False)
        output = tok.decode(gen[0][n_in:].cpu(), skip_special_tokens=True)
        verdict, detail = apply_check(output, case["check"])
        rows.append({"id": case["id"], "category": case["category"],
                     "lang": case["lang"], "verdict": verdict,
                     "detail": detail, "output": output,
                     "seconds": round(time.time() - t0, 2)})
        print(f"  [{i + 1:02d}/{len(cases)}] {case['id']:14s} {verdict:6s}",
              flush=True)
    return rows


def suite_toolcall(model, tok, device: str, sys_prompt: str | None) -> dict:
    from ayanami_distill.eval.harness import suite_toolcall as run
    trajs = [json.loads(line) for line in
             open(ROOT / "tests/fixtures/trajectories.jsonl", encoding="utf-8")]
    return run(model, tok, trajs, device, sys_prompt)


def suite_ppl_big(model, tok, device: str) -> dict:
    import torch
    import torch.nn.functional as F

    lines = [line.strip() for line in
             open(ROOT / "src/ayanami_distill/eval/data/heldout_big.txt",
                  encoding="utf-8") if line.strip()]
    lines = lines[::3]
    losses = []
    with torch.no_grad():
        for line in lines:
            ids = torch.tensor(
                [tok(line, add_special_tokens=False)["input_ids"][:48]],
                dtype=torch.long, device=device)
            if ids.shape[1] < 3:
                continue
            logits = model(ids).logits
            loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
            losses.append(float(loss))
    mean = sum(losses) / max(1, len(losses))
    return {"n_lines": len(losses), "mean_loss": round(mean, 4),
            "perplexity": round(math.exp(min(mean, 20.0)), 3),
            "lines": losses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--dtype", default="auto",
                    help="auto|bf16|fp16|fp32")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--suites", default="all",
                    help="comma list of persona,obedience,identity,injection,"
                         "devops,security,sandbox,toolcall,ppl, or all")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}
    if args.dtype == "auto":
        dtype = (torch.float16 if device == "cuda"
                 and not torch.cuda.is_bf16_supported() else torch.bfloat16)
    else:
        dtype = dtypes[args.dtype]

    wanted = ({s.strip() for s in args.suites.split(",")}
              if args.suites != "all" else set(SUITES) | {"toolcall", "ppl"})
    unknown = wanted - set(SUITES) - {"toolcall", "ppl"}
    if unknown:
        raise SystemExit(f"unknown suites: {sorted(unknown)}")

    materialize_dev()
    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys_prompt = None
    if args.system_prompt:
        sys_prompt = Path(args.system_prompt).read_text(encoding="utf-8")

    print(f"model={model_dir} device={device} dtype={dtype} "
          f"system_prompt={'yes' if sys_prompt else 'no'}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype, local_files_only=True).to(device)
    model.eval()

    doc: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(model_dir), "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "system_prompt": bool(sys_prompt), "seed": args.seed,
        "decoding": "greedy, enable_thinking=False", "dev_only": True,
        "suites": {},
    }
    for name in SUITES:
        if name not in wanted:
            continue
        path = DEV_DIR / f"{name}_dev.jsonl"
        cases = [json.loads(line) for line in
                 open(path, encoding="utf-8") if line.strip()]
        print(f"== {name} (n={len(cases)}, max_new={MAX_NEW.get(name, DEFAULT_MAX_NEW)}) ==",
              flush=True)
        rows = gen_suite(model, tok, cases, MAX_NEW.get(name, DEFAULT_MAX_NEW),
                         device, sys_prompt)
        doc["suites"][name] = {"summary": summarize(rows), "results": rows}

    if "toolcall" in wanted:
        print("== toolcall (10) ==", flush=True)
        doc["suites"]["toolcall"] = suite_toolcall(model, tok, device, sys_prompt)
    if "ppl" in wanted:
        print("== ppl_big (60 lines) ==", flush=True)
        doc["suites"]["ppl"] = suite_ppl_big(model, tok, device)
        print(f"  ppl={doc['suites']['ppl']['perplexity']}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
