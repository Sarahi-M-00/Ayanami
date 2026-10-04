#!/usr/bin/env python3
"""Phase 3/7 — Persona eval scorer.

Runs every case in persona/eval/cases.jsonl against a model, applies the
rule-based check declared in each case, and writes per-case results plus a
summary. Cases with check type "manual" are saved with verdict "review"
for Ling to judge.

Rule-based checks are directional signals, not final grades: a "fail" means
"needs review", especially for the untouched baseline.

Usage:
  PYTHONPATH=src .venv/bin/python -m ayanami_distill.eval.persona_eval \
    --cases persona/eval/cases.jsonl --out persona/eval/baseline0_results.jsonl \
    --summary persona/eval/baseline0_summary.json
  Optional: --system-prompt persona/system_prompt.md (Layer-1 run)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BULLET_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S")


def norm(s: str) -> str:
    return s.casefold()


def check_exact(output: str, value: str) -> tuple[str, str]:
    s = output.strip()
    if s.endswith("."):
        s = s[:-1].strip()
    ok = norm(s) == norm(value)
    return ("pass" if ok else "fail", f"got={s[:60]!r}")


def check_json(output: str) -> tuple[bool, str]:
    s = output.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        json.loads(s)
        return "pass", "parses as JSON"
    except Exception as exc:
        return "fail", f"not JSON: {exc}"


def apply_check(output: str, check: dict) -> tuple[str, str]:
    """Return (verdict, detail). Verdict in pass/fail/review."""
    t = check["type"]
    o = norm(output)
    if t == "contains_all":
        missing = [v for v in check["values"] if norm(v) not in o]
        return ("pass" if not missing else "fail", f"missing={missing}")
    if t == "contains_none":
        found = [v for v in check["values"] if norm(v) in o]
        return ("pass" if not found else "fail", f"forbidden_found={found}")
    if t == "canary_absent":
        # Injection canary: present in output ONLY if the injection was followed.
        found = [v for v in check["values"] if norm(v) in o]
        return ("pass" if not found else "fail", f"canary_followed={found}")
    if t == "regex":
        m = re.search(check["pattern"], output)
        return ("pass" if m else "fail", f"pattern={check['pattern']!r}")
    if t == "exact":
        return check_exact(output, check["value"])
    if t == "json_valid":
        ok, detail = check_json(output)
        return ("pass" if ok else "fail", detail)
    if t == "line_bullets":
        n = sum(1 for line in output.splitlines() if BULLET_RE.match(line))
        return ("pass" if n == check["count"] else "fail",
                f"bullets={n} expected={check['count']}")
    if t == "manual":
        return ("review", "manual review by Ling")
    if t == "all":
        # Composite: pass iff every sub-check passes (e.g. finding present
        # AND canary absent). Used by sandbox canary tasks.
        details = []
        for sub in check.get("checks", []):
            v, d = apply_check(output, sub)
            details.append(f"{sub.get('type')}:{v}({d})")
            if v != "pass":
                return ("fail", "; ".join(details))
        return ("pass", "; ".join(details))
    return ("review", f"unknown check type {t!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--model", default="student_base")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--start", type=int, default=0,
                    help="first case index (10.1 sharding for the 10-min CPU rule)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max cases to run from --start")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    root = Path(__file__).resolve().parents[3]
    model_dir = (root / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    sys_prompt = None
    if args.system_prompt:
        sys_prompt = Path(args.system_prompt).read_text(encoding="utf-8")

    cases = [json.loads(line) for line in
             open(args.cases, encoding="utf-8") if line.strip()]
    if args.limit is not None:
        cases = cases[args.start:args.start + args.limit]
    elif args.start:
        cases = cases[args.start:]
    print(f"cases: {len(cases)} (start={args.start} limit={args.limit}), "
          f"model: {model_dir}, system_prompt: {'yes' if sys_prompt else 'no'}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=True)
    model.eval()

    results = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases):
            messages = list(case["messages"])
            if sys_prompt:
                messages = [{"role": "system", "content": sys_prompt}] + messages
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tok(prompt_text, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            dt = time.time() - t0
            n_new = gen.shape[1] - n_in
            output = tok.decode(gen[0][n_in:], skip_special_tokens=True)
            verdict, detail = apply_check(output, case["check"])
            rec = {"id": case["id"], "category": case["category"],
                   "lang": case["lang"], "check_type": case["check"]["type"],
                   "verdict": verdict, "detail": detail, "output": output,
                   "input_tokens": n_in, "output_tokens": n_new,
                   "seconds": round(dt, 2)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            print(f"[{i + 1:02d}/{len(cases)}] {case['id']:14s} {verdict:6s} "
                  f"{detail[:70]}", flush=True)

    summary: dict = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "cases": len(cases), "model": str(model_dir),
                     "system_prompt": bool(sys_prompt), "seed": args.seed,
                     "max_new_tokens": args.max_new_tokens,
                     "by_category": {}, "by_lang": {}, "overall": {}}
    buckets: dict = {"overall:all": []}
    for r in results:
        buckets.setdefault(f"cat:{r['category']}", []).append(r["verdict"])
        buckets.setdefault(f"lang:{r['lang']}", []).append(r["verdict"])
        buckets["overall:all"].append(r["verdict"])
    for name, verdicts in buckets.items():
        kind, label = name.split(":", 1)
        entry = {v: verdicts.count(v) for v in ("pass", "fail", "review")}
        entry["pass_rate_auto"] = round(
            entry["pass"] / max(1, entry["pass"] + entry["fail"]), 4)
        if kind == "overall":
            summary["overall"] = entry
        elif kind == "cat":
            summary["by_category"][label] = entry
        else:
            summary["by_lang"][label] = entry
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
