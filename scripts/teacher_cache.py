#!/usr/bin/env python3
"""Stage 10.3 — Teacher-side cache builder (runs on REMOTE GPU, not the laptop).

Two subcommands:
  select  stratified pilot/bulk prompt selection from the v1 pool.
  cache   generate (sampling) + scrub + verify + teacher-forced top-k scoring.

Remote requirements: CUDA GPU, bitsandbytes (installed by setup_remote.sh),
network for the one-time teacher download. Single GPU only (world size 1).
Shards are idempotent by record ID: restarts skip finished IDs.

Teacher: Qwen/Qwen3-8B (Apache-2.0). 4-bit NF4, fp16 compute (no bf16 on
T4/P100). Fingerprint MUST equal 563a701b… or the run aborts.
Non-thinking sampling per the model card: T=0.7, top_p=0.8, top_k=20, min_p=0.

Usage (on the GPU machine, from the repo root):
  .venv/bin/python scripts/teacher_cache.py select --pool data/processed/prompts_v1.jsonl --n 50 --out data/processed/pilot_50.jsonl
  .venv/bin/python scripts/teacher_cache.py cache --prompts data/processed/pilot_50.jsonl --out data/teacher_cache/qwen-qwen3-8b/ --shard 0 --shard-size 100
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDENT_FINGERPRINT = "563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa"
TEACHER_ID = "Qwen/Qwen3-8B"
TEACHER_LICENSE = "Apache-2.0"
GEN_PARAMS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
              "max_tokens": 512, "do_sample": True}
TOPK = 32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_of_tokenizer(tok) -> str:
    """Audit-identical fingerprint (Phase 1 method), via a temp save.

    NOTE: get_vocab() must NOT be used here — it merges/renames added
    tokens, producing a different hash for the same tokenizer (caught by
    test: 51e63d0d… vs the canonical 563a701b…). Saving and reading the
    tokenizer.json reproduces the audit computation exactly.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        with open(Path(tmp) / "tokenizer.json", encoding="utf-8") as fh:
            tj = json.load(fh)
    src = json.dumps({"vocab": sorted(tj["model"]["vocab"].items()),
                      "added_tokens": tj.get("added_tokens", [])},
                     sort_keys=True)
    return hashlib.sha256(src.encode()).hexdigest()


def check_fingerprint(tok) -> str:
    fp = fingerprint_of_tokenizer(tok)
    if fp != STUDENT_FINGERPRINT:
        raise SystemExit(f"ABORT: teacher fingerprint {fp} != student "
                         f"{STUDENT_FINGERPRINT} (logit_kd would be invalid)")
    return fp


def load_teacher(model_id: str = TEACHER_ID, revision: str | None = None):
    """Load Qwen3-8B NF4/fp16 on CUDA. Raises a clear error without GPU/bitsandbytes."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA GPU (teacher_cache runs on remote GPU only)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(f"ABORT: missing dependency: {exc}")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        raise SystemExit("ABORT: bitsandbytes is required on the GPU machine "
                         "(setup_remote.sh installs it)")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    fp = check_fingerprint(tok)
    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=qconf,
        dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tok, fp


def gen_one(model, tok, messages: list[dict], seed: int) -> str:
    import torch
    torch.manual_seed(seed)
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=GEN_PARAMS["max_tokens"],
            do_sample=True, temperature=GEN_PARAMS["temperature"],
            top_p=GEN_PARAMS["top_p"], top_k=GEN_PARAMS["top_k"],
            min_p=GEN_PARAMS["min_p"], pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def score_topk(model, tok, messages: list[dict], completion: str, k: int = TOPK) -> dict:
    """Teacher-forced top-k at T=1 over ASSISTANT tokens only.

    Never reuses sampling-modified scores: this is a separate forward pass
    over the final sequence. Returns {positions, k, ids, logprobs, tail_mass}.
    """
    import torch
    import torch.nn.functional as F
    prefix = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    full_msgs = messages + [{"role": "assistant", "content": completion}]
    full = tok.apply_chat_template(full_msgs, tokenize=False,
                                   add_generation_prompt=False)
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert full_ids[:len(pre_ids)] == pre_ids, "prefix instability in scoring"
    positions = list(range(len(pre_ids), len(full_ids)))
    out = {"positions": positions, "k": k, "ids": [], "logprobs": [], "tail_mass": []}
    if not positions:
        return out
    with torch.no_grad():
        logits = model(torch.tensor([full_ids], device=model.device)).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    for t in positions:
        lp = logp[t - 1] if t > 0 else logp[0]
        vals, idx = torch.topk(lp, k)
        top = vals.exp()
        out["ids"].append(idx.tolist())
        out["logprobs"].append(vals.tolist())
        out["tail_mass"].append(float((1.0 - top.sum()).clamp_min(0.0)))
    return out


def process_prompt(prompt: dict, gen_fn, score_fn, scrub_fn, verify_fn,
                   meta: dict) -> tuple[dict | None, str]:
    """Generate -> scrub -> verify -> score. Returns (record|None, reason)."""
    from ayanami_distill.tools.validate import extract_tool_calls
    seed = meta["seed"] + int(prompt["id"].split("-")[-1])
    completion = gen_fn(prompt["messages"], seed)
    cleaned, hits = scrub_fn(completion)
    verdict, detail = verify_fn(prompt, cleaned)
    if verdict != "pass":
        return None, f"verifier:{verdict}:{detail}"
    # second scrub pass: unfixable residue -> drop
    _, hits2 = scrub_fn(cleaned)
    if hits2:
        return None, f"scrub-residue:{hits2}"
    topk = score_fn(prompt["messages"], cleaned)
    calls, errs = extract_tool_calls(cleaned)
    if errs:
        return None, f"toolparse:{errs}"
    rec = {
        "id": prompt["id"],
        "teacher_id": meta["teacher_id"],
        "teacher_revision": meta["revision"],
        "tokenizer_fingerprint": meta["fingerprint"],
        "license": TEACHER_LICENSE,
        "tos_allows_distillation": True,
        "domain": prompt["category"],
        "messages": prompt["messages"],
        "student_messages": prompt["student_messages"],
        "completion": cleaned,
        "reasoning": None,
        "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls],
        "tool_results": [],
        "logprobs_topk": topk,
        "hidden_ref": None,
        "gen_params": {**GEN_PARAMS, "seed": seed},
        "verifier": prompt.get("verifier"),
        "split": "train",
        "verified_success": None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return rec, f"kept scrub_hits={len(hits)}"


def cmd_select(args) -> int:
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8") if line.strip()]
    rng = random.Random(args.seed)
    by_cat: dict[str, list] = {}
    for p in pool:
        by_cat.setdefault(p["category"], []).append(p)
    per = max(1, args.n // len(by_cat))
    picked = []
    for cat in sorted(by_cat):
        items = list(by_cat[cat])
        rng.shuffle(items)
        picked.extend(items[:per])
    rng.shuffle(picked)
    picked = picked[:args.n]
    with open(args.out, "w", encoding="utf-8") as fh:
        for p in picked:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    cats: dict[str, int] = {}
    for p in picked:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"selected {len(picked)}: {cats} -> {args.out}")
    return 0


def build_manifest(outdir: Path, shard: int, kept: int, dropped: dict,
                   meta: dict, stats: dict) -> dict:
    """Pure manifest assembly (testable without GPU)."""
    import torch
    import transformers
    shard_path = outdir / f"shard_{shard:03d}.jsonl"
    manifest = {
        "teacher_id": TEACHER_ID, "revision": meta.get("revision", "main"),
        "fingerprint": meta["fingerprint"], "gen_params": GEN_PARAMS, "topk": TOPK,
        "shard": shard, "shard_file": shard_path.name,
        "sha256": sha256_file(shard_path),
        "n_kept": kept, "rejection": dropped,
        "gpu": stats.get("gpu", "unknown"), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gen_tokens_per_sec": stats.get("gen_tps"),
        "vram_peak_gb": stats.get("vram_gb"),
        "wall_seconds": stats.get("wall_s"),
    }
    with open(outdir / f"manifest_{shard:03d}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def cmd_cache(args) -> int:
    import torch
    model, tok, fp = load_teacher(args.model, args.revision)
    gpu = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")
               if line.strip()]
    shard_prompts = prompts[args.shard * args.shard_size:(args.shard + 1) * args.shard_size]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    shard_path = outdir / f"shard_{args.shard:03d}.jsonl"
    done = set()
    if shard_path.is_file():
        for line in open(shard_path, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"shard {args.shard}: {len(shard_prompts)} prompts, {len(done)} done, GPU {gpu}",
          flush=True)

    from ayanami_distill.data.verifiers import run_verifier
    from ayanami_distill.teachers.scrub import scrub_completion
    meta = {"teacher_id": TEACHER_ID, "revision": args.revision or "main",
            "fingerprint": fp, "seed": args.seed}
    kept, dropped = 0, {}
    t0 = time.time()
    gen_tok, gen_s = 0, 0.0

    def verify_fn(prompt, cleaned):
        vf = prompt.get("verifier") or {}
        if not vf:
            return "pass", "no verifier (replay)"
        return run_verifier(vf["name"], cleaned, {k: v for k, v in vf.items()
                                                  if k != "name"})

    for p in shard_prompts:
        if p["id"] in done:
            continue
        t1 = time.time()
        rec, reason = process_prompt(
            p,
            lambda msgs, seed: gen_one(model, tok, msgs, seed),
            lambda msgs, comp: score_topk(model, tok, msgs, comp),
            scrub_completion, verify_fn, meta)
        gen_s += time.time() - t1
        if rec is None:
            key = reason.split(":")[0]
            dropped[key] = dropped.get(key, 0) + 1
            print(f"  DROP {p['id']}: {reason}", flush=True)
            continue
        gen_tok += len(tok(rec["completion"], add_special_tokens=False)["input_ids"])
        with open(shard_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        print(f"  kept {p['id']} ({kept})", flush=True)

    stats = {"gpu": gpu,
             "gen_tps": round(gen_tok / max(gen_s, 1e-9), 2),
             "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
             "wall_s": round(time.time() - t0, 1)}
    meta = {"revision": args.revision or "main", "fingerprint": fp}
    manifest = build_manifest(outdir, args.shard, kept, dropped, meta, stats)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "gen_params"}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Teacher cache builder (remote GPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--pool", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--prompts", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=TEACHER_ID)
    c.add_argument("--revision", default=None)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--shard-size", type=int, default=100)
    c.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.cmd == "select":
        return cmd_select(args)
    return cmd_cache(args)


if __name__ == "__main__":
    sys.exit(main())
