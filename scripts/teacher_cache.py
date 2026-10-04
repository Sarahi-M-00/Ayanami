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
import zlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ayanami_distill.data.verifiers import run_verifier  # noqa: E402
from ayanami_distill.teachers.scrub import scrub_completion  # noqa: E402
from ayanami_distill.tools.validate import extract_tool_calls  # noqa: E402

STUDENT_FINGERPRINT = "563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa"
TEACHER_ID = "Qwen/Qwen3-8B"
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TEACHER_LICENSE = "Apache-2.0"
GEN_PARAMS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
              "max_tokens": 512, "do_sample": True}
TOPK = 32
EOS_IDS = {151645, 151643}


class PrefixUnstableError(RuntimeError):
    """Assistant-span prefix did not align; the record is dropped, not fatal."""


def resolve_teacher_revision(model_id: str = TEACHER_ID) -> str:
    """Resolve the pinned commit SHA from the Hub (no weights downloaded)."""
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha


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
    # Single GPU always (world size 1): never let accelerate split layers.
    # Two parallel processes -> CUDA_VISIBLE_DEVICES=0 / =1 per process.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=qconf,
        dtype=torch.float16, device_map={"": 0})
    model.eval()
    return model, tok, fp


def registry_tools():
    """Registry schemas as OpenAI-style tool defs for the chat template."""
    from ayanami_distill.tools.catalog import build_default_registry
    reg = build_default_registry()
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.arguments}}
            for t in (reg.get(n) for n in reg.names())]


def gen_one(model, tok, messages: list[dict], seed: int,
            tools: list[dict] | None = None) -> tuple[str, bool]:
    """Generate one completion. Returns (text, finished).

    finished=False (hit max_new_tokens without EOS, or empty) means the
    record must be dropped: a cut turn would teach arbitrary stopping.
    """
    import torch
    torch.manual_seed(seed)
    kwargs = {} if tools is None else {"tools": tools}
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False, **kwargs)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=GEN_PARAMS["max_tokens"],
            do_sample=True, temperature=GEN_PARAMS["temperature"],
            top_p=GEN_PARAMS["top_p"], top_k=GEN_PARAMS["top_k"],
            min_p=GEN_PARAMS["min_p"], pad_token_id=tok.eos_token_id)
    new_ids = gen[0][inputs["input_ids"].shape[1]:].tolist()
    eos = set(EOS_IDS) | {tok.eos_token_id}
    finished = bool(new_ids) and new_ids[-1] in eos
    return tok.decode(new_ids, skip_special_tokens=True), finished


def score_topk(model, tok, messages: list[dict], completion: str, k: int = TOPK) -> dict:
    """Teacher-forced top-k at T=1 over ASSISTANT tokens only.

    Never reuses sampling-modified scores: this is a separate forward pass
    over the final sequence. Returns {positions, k, ids, logprobs, tail_mass}
    with logprobs rounded to 4 decimals (JSON size) and float32 math.

    Raises PrefixUnstableError instead of aborting the shard. Drops the
    single template-added trailing newline after the final <|im_end|>
    (the model never generated it; recorded decision 2026-10-04).
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
    if full_ids[:len(pre_ids)] != pre_ids:
        raise PrefixUnstableError("assistant-span prefix misaligned")
    positions = list(range(len(pre_ids), len(full_ids)))
    # Drop template-added trailing newline after the final im_end.
    newline_ids = tok("\n", add_special_tokens=False)["input_ids"]
    if positions and len(newline_ids) == 1 and full_ids[-1] == newline_ids[0] \
            and not completion.endswith("\n") and full_ids[-2] == 151645:
        positions = positions[:-1]
    out = {"positions": positions, "k": k, "ids": [], "logprobs": [], "tail_mass": []}
    if not positions:
        return out
    with torch.no_grad():
        logits = model(torch.tensor([full_ids], device=model.device)).logits[0]
        rows = logits[[t - 1 for t in positions]].float()
        del logits
        logp = F.log_softmax(rows, dim=-1)
        del rows
        vals, idx = torch.topk(logp, k, dim=-1)
        tail = (1.0 - vals.exp().sum(-1)).clamp_min(0.0)
    out["ids"] = idx.tolist()
    out["logprobs"] = [[round(float(x), 4) for x in row] for row in vals.tolist()]
    out["tail_mass"] = [round(float(x), 4) for x in tail.tolist()]
    return out


def process_prompt(prompt: dict, gen_fn, score_fn, scrub_fn, verify_fn,
                   meta: dict) -> tuple[dict | None, str]:
    """Generate -> finish-check -> scrub -> verify -> score. Returns (record|None, reason).

    Every drop reason carries a completion excerpt (first 300 chars, flattened)
    so rejected records stay eye-reviewable in rejected_<shard>.jsonl.
    """
    def drop(reason: str, text: str) -> tuple[None, str]:
        flat = " ".join(text.split())[:300]
        return None, f"{reason} | excerpt={flat}"

    seed = meta["seed"] + (zlib.crc32(prompt["id"].encode()) % 100000)
    tools = registry_tools() if prompt.get("category") == "tool_use" else None
    completion, finished = gen_fn(prompt["messages"], seed, tools)
    if not finished:
        return drop("truncated", completion)
    if "<think>" in completion or "</think>" in completion:
        return drop("think-leak", completion)
    cleaned, hits = scrub_fn(completion)
    verdict, detail = verify_fn(prompt, cleaned)
    if verdict != "pass":
        return drop(f"verifier:{verdict}:{detail}", cleaned)
    # second scrub pass: unfixable residue -> drop
    _, hits2 = scrub_fn(cleaned)
    if hits2:
        return drop(f"scrub-residue:{hits2}", cleaned)
    try:
        topk = score_fn(prompt["messages"], cleaned)
    except PrefixUnstableError:
        return drop("prefix-unstable", cleaned)
    calls, errs = extract_tool_calls(cleaned)
    if errs:
        return drop(f"toolparse:{errs}", cleaned)
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
        "split": prompt.get("split", "train"),
        "verified_success": True if prompt["category"] == "tool_use" else None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return rec, f"kept scrub_hits={len(hits)}"


def cmd_select(args) -> int:
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8") if line.strip()]
    rng = random.Random(args.seed)
    by_cat: dict[str, list] = {}
    for p in pool:
        by_cat.setdefault(p["category"], []).append(p)
    per, rem = divmod(args.n, len(by_cat))
    per = max(1, per)
    picked = []
    for i, cat in enumerate(sorted(by_cat)):
        items = list(by_cat[cat])
        rng.shuffle(items)
        # Distribute the remainder round-robin so we return exactly n.
        picked.extend(items[:per + (1 if i < rem else 0)])
    rng.shuffle(picked)
    picked = picked[:args.n]
    assert len(picked) == args.n, f"select returned {len(picked)} != {args.n}"
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
        "timing": stats.get("timing", {}),
        "vram_peak_gb": stats.get("vram_gb"),
        "wall_seconds": stats.get("wall_s"),
    }
    with open(outdir / f"manifest_{shard:03d}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def repair_shard(shard_path: Path) -> tuple[set[str], set[str]]:
    """Recover from a mid-write crash: drop the truncated tail, keep good lines.

    Returns (kept_ids, rejected_ids). Rejected IDs (from rejected_<shard>.jsonl)
    are treated as done: they were already judged, never regenerated.
    """
    good: list[dict] = []
    if shard_path.is_file():
        for line in open(shard_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated tail after a crash; stop, discard rest
        shard_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in good), encoding="utf-8")
    rej_path = shard_path.parent / shard_path.name.replace("shard_", "rejected_")
    rejected: set[str] = set()
    if rej_path.is_file():
        for line in open(rej_path, encoding="utf-8"):
            if line.strip():
                try:
                    rejected.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    break
    return {r["id"] for r in good}, rejected


def cmd_preflight(args) -> int:
    """No GPU, no weights: imports, tokenizer, fingerprint, prompt checks."""
    import torch
    import transformers
    from transformers import AutoTokenizer
    print(f"python ok; torch={torch.__version__} "
          f"transformers={transformers.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    fp = check_fingerprint(tok)
    print(f"tokenizer ok; fingerprint={fp}", flush=True)
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")
               if line.strip()]
    bad = 0
    for p in prompts:
        for field in ("messages", "student_messages", "category"):
            if field not in p:
                print(f"  MISSING {field} in {p.get('id')}")
                bad += 1
        try:
            tok.apply_chat_template(p["messages"], tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
        except Exception as exc:
            print(f"  RENDER FAIL {p.get('id')}: {exc!r}")
            bad += 1
    print(f"preflight: {len(prompts)} prompts, {bad} problems", flush=True)
    return 0 if bad == 0 else 2


def verify_with_lang(prompt: dict, cleaned: str) -> tuple[str, str]:
    """Verifier dispatch with automatic language gate for persona rows.

    Persona/identity answers must satisfy BOTH their named verifier and the
    response language (the pilot caught an ES prompt answered in English).
    """
    from ayanami_distill.data.verifiers import VERIFIERS
    vf = prompt.get("verifier") or {}
    if not vf:
        return "pass", "no verifier (replay)"
    params = {k: v for k, v in vf.items() if k != "name"}
    if vf["name"] == "lang_match":
        params.setdefault("lang", prompt.get("lang", "en"))
    verdict, detail = run_verifier(vf["name"], cleaned, params)
    if verdict != "pass":
        return verdict, detail
    if prompt.get("category") == "persona":
        lv, ld = VERIFIERS["lang_match"](
            cleaned, {"lang": prompt.get("lang", "en")})
        if lv != "pass":
            return "fail", f"lang_gate:{ld}"
    return "pass", detail


def run_shard(model, tok, fp, prompts: list[dict], outdir: Path, shard: int,
              seed: int, revision: str, gen_fn=None, score_fn=None) -> dict:
    """Run one shard (model already loaded). Returns manifest dict.

    gen_fn/score_fn injectable for CPU tests; default to the real ones.
    """
    import torch
    gen_fn = gen_fn or (lambda msgs, seed, tools: gen_one(model, tok, msgs, seed, tools))
    score_fn = score_fn or (lambda msgs, comp: score_topk(model, tok, msgs, comp))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram = (round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else 0.0)
    shard_path = outdir / f"shard_{shard:03d}.jsonl"
    rej_path = outdir / f"rejected_{shard:03d}.jsonl"
    done, rejected = repair_shard(shard_path)
    meta = {"teacher_id": TEACHER_ID, "revision": revision,
            "fingerprint": fp, "seed": seed}
    cat_time: dict[str, float] = {}
    cat_kept: dict[str, int] = {}
    rej_sec = 0.0
    t0 = time.time()

    def verify_fn(prompt, cleaned):
        return verify_with_lang(prompt, cleaned)

    for p in prompts:
        if p["id"] in done or p["id"] in rejected:
            continue
        t1 = time.time()
        rec, reason = process_prompt(
            p, gen_fn, score_fn, scrub_completion, verify_fn, meta)
        dt = time.time() - t1
        cat = p.get("category", "?")
        if rec is None:
            rej_sec += dt
            with open(rej_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": p["id"], "reason": reason}) + "\n")
            print(f"  DROP {p['id']}: {reason[:160]}", flush=True)
            continue
        cat_time[cat] = cat_time.get(cat, 0.0) + dt
        cat_kept[cat] = cat_kept.get(cat, 0) + 1
        with open(shard_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  kept {p['id']} ({sum(cat_kept.values())})", flush=True)

    # Honest totals from FILES, not from this run alone.
    kept_ids, _ = repair_shard(shard_path)
    dropped: dict[str, int] = {}
    if rej_path.is_file():
        for line in open(rej_path, encoding="utf-8"):
            if line.strip():
                key = json.loads(line)["reason"].split(":")[0]
                dropped[key] = dropped.get(key, 0) + 1
    timing = {"sec_per_kept_by_category":
              {c: round(cat_time.get(c, 0.0) / max(1, cat_kept.get(c, 0)), 2)
               for c in cat_time},
              "rejected_seconds": round(rej_sec, 1)}
    stats = {"gpu": gpu, "vram_gb": vram,
             "wall_s": round(time.time() - t0, 1), "timing": timing}
    return build_manifest(outdir, shard, len(kept_ids), dropped,
                          {"revision": revision, "fingerprint": fp}, stats)


def cmd_cache(args) -> int:
    import torch
    if args.revision != TEACHER_REVISION and not args.force_revision:
        raise SystemExit(f"ABORT: revision {args.revision} != pinned {TEACHER_REVISION} "
                         f"(pass --force-revision to override explicitly)")
    model, tok, fp = load_teacher(args.model, args.revision)
    gpu = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")
               if line.strip()]
    bad_splits = {p.get("split", "train") for p in prompts} - {"train"}
    if bad_splits:
        raise SystemExit(f"ABORT: prompt file has non-train splits: {bad_splits}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    lo, hi = ((int(x) for x in args.shards.split("-")) if args.shards
              else (args.shard, args.shard))
    for shard in range(lo, hi + 1):
        shard_prompts = prompts[shard * args.shard_size:(shard + 1) * args.shard_size]
        if not shard_prompts:
            print(f"shard {shard}: empty, skipping", flush=True)
            continue
        print(f"shard {shard}: {len(shard_prompts)} prompts, GPU {gpu}", flush=True)
        manifest = run_shard(model, tok, fp, shard_prompts, outdir, shard,
                             args.seed, args.revision)
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
    p = sub.add_parser("preflight",
                       help="no GPU/weights: imports, tokenizer, fingerprint, prompts")
    p.add_argument("--prompts", required=True)
    p.add_argument("--model", default=TEACHER_ID)
    p.add_argument("--revision", default=TEACHER_REVISION)
    c = sub.add_parser("cache")
    c.add_argument("--prompts", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=TEACHER_ID)
    c.add_argument("--revision", default=TEACHER_REVISION)
    c.add_argument("--force-revision", action="store_true",
                   help="allow a non-pinned revision explicitly")
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--shards", default=None,
                   help="range like 0-9: loop in one process, one model load")
    c.add_argument("--shard-size", type=int, default=100)
    c.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    return cmd_cache(args)


if __name__ == "__main__":
    sys.exit(main())
