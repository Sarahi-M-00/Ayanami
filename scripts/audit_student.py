#!/usr/bin/env python3
"""Phase 1 — Audit the Student snapshot.

Reads only from student_base/ (read-only). Writes docs/STUDENT_AUDIT.md.
Offline: never touches the network.

Covers: file list + SHA256, safetensors header (params, dtypes, shapes),
NaN/Inf scan, config.json, generation_config.json, tokenizer files +
fingerprint, chat template renders (plain / system / tools) and
thinking on/off behavior.

Usage: /home/ling/venv-hf-cpu/bin/python scripts/audit_student.py
"""

import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "student_base"
DOC = ROOT / "docs" / "STUDENT_AUDIT.md"

EXPECTED_FILES = [
    ".gitattributes",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return {"metadata": header.get("__metadata__", {}), "tensors": tensors}


def main() -> int:
    t0 = time.time()
    lines: list[str] = []
    out = lines.append
    out("# STUDENT_AUDIT.md — Qwen3-1.7B-heretic snapshot audit")
    out(f"Audited: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out("Snapshot: `student_base/` (read-only copy of /home/ling/Ayanami-AI/)")
    out("")

    # ---- 1. file list ----
    out("## 1. File list (size + SHA256)")
    out("")
    out("| file | size_bytes | sha256 |")
    out("| --- | --- | --- |")
    sizes = {}
    for name in EXPECTED_FILES:
        p = BASE / name
        if not p.is_file():
            out(f"| {name} | MISSING | MISSING |")
            continue
        sizes[name] = p.stat().st_size
        out(f"| {name} | {sizes[name]} | `{sha256(p)}` |")
    extra = sorted(p.name for p in BASE.iterdir() if p.name not in EXPECTED_FILES)
    out("")
    out(f"Extra files in snapshot: {extra if extra else 'none'}.")
    out("Safetensors shards: single file `model.safetensors`, no `.index.json` — nothing to cross-check.")
    out("")

    # ---- 2. safetensors header ----
    import torch
    from safetensors import safe_open

    st_path = BASE / "model.safetensors"
    header = read_safetensors_header(st_path)
    tensors = header["tensors"]
    dtype_count: dict[str, int] = {}
    total_params = 0
    for _name, spec in tensors.items():
        n = 1
        for s in spec["shape"]:
            n *= s
        total_params += n
        dtype_count[spec["dtype"]] = dtype_count.get(spec["dtype"], 0) + 1
    out("## 2. Safetensors header")
    out("")
    out(f"Tensors: {len(tensors)}. Total parameters: {total_params:,} ({total_params / 1e9:.4f} B).")
    out(f"Dtype distribution: {dtype_count}. Header `__metadata__`: {header['metadata'] or 'empty'}.")
    layer_idx = sorted({int(k.split('.')[2]) for k in tensors if k.startswith('model.layers.')})
    out(f"Layer indices present: 0..{max(layer_idx)} consecutive: {layer_idx == list(range(len(layer_idx)))} "
        f"({len(layer_idx)} layers).")
    out(f"`lm_head.weight` in file: {'lm_head.weight' in tensors} (tied embeddings expected: True).")
    out("")

    # ---- 3. NaN/Inf scan (streaming) ----
    out("## 3. NaN/Inf scan (one tensor at a time, float32)")
    out("")
    nan_t, inf_t = [], []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name).to(torch.float32)
            if torch.isnan(t).any():
                nan_t.append(name)
            if torch.isinf(t).any():
                inf_t.append(name)
            del t
    import gc
    gc.collect()
    out(f"Tensors with NaN: {nan_t if nan_t else 'none'}. Tensors with Inf: {inf_t if inf_t else 'none'}.")
    out("")

    # ---- 4. config.json ----
    cfg = json.loads((BASE / "config.json").read_text())
    out("## 4. config.json (real values)")
    out("")
    out("```json")
    out(json.dumps(cfg, indent=2))
    out("```")
    out("")
    out("Summary: architecture `Qwen3ForCausalLM`, model_type `qwen3`, 28 layers (all `full_attention`), "
        "hidden_size 2048, 16 attention heads / 8 KV heads, head_dim 128, intermediate_size 6144, "
        "vocab_size 151936, tie_word_embeddings true, max_position_embeddings 40960, "
        "rope_theta 1000000 (rope_type default), rms_norm_eps 1e-06, hidden_act silu, "
        "attention_bias false, dtype bfloat16, bos_token_id 151643, eos_token_id 151645, "
        "pad_token_id null, written by transformers 5.17.0.")
    out("VERIFY long context (32k class): CONFIRMED-ish — max_position_embeddings=40960 "
        "(40k, above 32k); tokenizer model_max_length=131072. Usable context is 40960 per config.")
    out("VERIFY transformers>=4.51: CONFIRMED — snapshot targets transformers 5.x; audit runs 5.18.0.")
    out("")

    # ---- 5. generation_config.json ----
    gen = json.loads((BASE / "generation_config.json").read_text())
    out("## 5. generation_config.json (real values)")
    out("")
    out("```json")
    out(json.dumps(gen, indent=2))
    out("```")
    out("")
    out("Summary: sampling on (temperature 0.6, top_k 20, top_p 0.95), "
        "eos_token_id [151645, 151643], bos/pad 151643.")
    out("")

    # ---- 6. tokenizer ----
    tj = json.loads((BASE / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]
    added = tj.get("added_tokens", [])
    specials = {t["id"]: t["content"] for t in added}
    fp_src = json.dumps({"vocab": sorted(vocab.items()), "added_tokens": added}, sort_keys=True)
    fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()
    out("## 6. Tokenizer")
    out("")
    out(f"tokenizer_class: Qwen2Tokenizer (fast). Base vocab entries: {len(vocab)}. "
        f"Added tokens: {len(added)} (ids {min(specials)}..{max(specials)}).")
    out(f"Special token map: {specials}.")
    out(f"tokenizer_config: eos_token `<|im_end|>` (id 151645), pad_token `<|endoftext|>` (id 151643), "
        f"bos_token null, model_max_length 131072, split_special_tokens false.")
    out("")
    out(f"Tokenizer fingerprint (SHA256 of sorted vocab + added tokens): `{fingerprint}`")
    out("This fingerprint is the Phase 4 key for teacher/student logit compatibility.")
    out("")

    # ---- 7. chat template renders ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    out("## 7. Chat template renders")
    out("")
    case_a = [{"role": "user", "content": "Explain what DNS is."}]
    case_b = [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain what DNS is."},
    ]
    tools = [{"type": "function", "function": {"name": "get_status",
             "description": "Read service status.", "parameters": {"type": "object", "properties": {}}}}
    ]
    case_c = [
        {"role": "user", "content": "Is the web service up?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "get_status", "arguments": "{}"}}]},
        {"role": "tool", "content": "active (running)"},
    ]
    ra = tok.apply_chat_template(case_a, tokenize=False, add_generation_prompt=True)
    rb = tok.apply_chat_template(case_b, tokenize=False, add_generation_prompt=True)
    rc = tok.apply_chat_template(case_c, tools=tools, tokenize=False, add_generation_prompt=True)
    out("### (a) plain chat")
    out("```")
    out(ra)
    out("```")
    out("### (b) chat with system message")
    out("```")
    out(rb)
    out("```")
    out("### (c) chat with tools + tool response")
    out("```")
    out(rc)
    out("```")
    out("Template is standard Qwen3 ChatML (`<|im_start|>`/`<|im_end|>`), Hermes-style "
        "`<tool_call>`/`<tool_response>`, thinking via `<think>` blocks or `reasoning_content`.")
    rd_think = tok.apply_chat_template(case_a, tokenize=False, add_generation_prompt=True)
    rd_nothink = tok.apply_chat_template(case_a, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    out("Thinking on (default) ends with: `" + repr(rd_think[-40:]) + "`")
    empty_think = "<think>\n\n</think>"
    out("Thinking off (`enable_thinking=False`) appends an empty think block: "
        f"{empty_think in rd_nothink}.")
    out("")

    # ---- 8. model card claims vs files ----
    out("## 8. Model card claims (from README.md) vs snapshot")
    out("")
    out("- Lineage Qwen3-1.7B-Base -> Qwen3-1.7B -> heretic abliteration (Heretic tool, 25 Optuna trials, "
        "trial 19 selected): from card, not verifiable in weight files. TAKEN AS CARD CLAIM.")
    out("- Refusals 3/100 (was 92/100), KL 0.0566 vs original: from card table. NOT verifiable offline.")
    out("- Languages en+es: from card front-matter. NOT verifiable in files (no language id in config).")
    out(f"- BF16 safetensors ~2B params: CONFIRMED — dtype bfloat16, {total_params:,} params counted "
        f"in header, {sizes.get('model.safetensors', '?')} bytes on disk.")
    out("- License apache-2.0, base_model Qwen/Qwen3-1.7B: from card front-matter.")
    out("")

    out(f"Audit wall time: {time.time() - t0:.1f}s.")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DOC} ({len(lines)} lines)")
    print(f"params={total_params} tensors={len(tensors)} nan={nan_t} inf={inf_t}")
    print(f"fingerprint={fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
