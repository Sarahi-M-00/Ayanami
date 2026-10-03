"""Minimal deterministic LoRA trainer (Phase 5, text_sft mode).

Scope today: response-based SFT on canonical records with assistant-only
loss masks. Other modes (logit_kd, cross_tokenizer, hidden_kd) have configs
and loss modules but no training entry point until their data exists.

Features: seeded runs, gradient accumulation, linear-warmup cosine
schedule, gradient checkpointing, eval loss + perplexity on a held-out
split (drift tracking), adapter + optimizer checkpointing with resume,
append-only metrics JSONL, and run metadata (config, versions, data
hashes, mix provenance) under runs/<run_id>/.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.mix import build_mix, load_jsonl
from ..losses.text_sft import IGNORE_INDEX, assistant_mask_from_messages

ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_messages(tokenizer, messages: list[dict], max_len: int) -> dict[str, list[int]]:
    ids, mask = assistant_mask_from_messages(tokenizer, messages)
    if len(ids) > max_len:  # keep the tail: the response matters most
        ids, mask = ids[-max_len:], mask[-max_len:]
    labels = [i if m else IGNORE_INDEX for i, m in zip(ids, mask)]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def collate(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    maxlen = max(len(e["input_ids"]) for e in examples)
    out = {}
    for key, pad in (("input_ids", pad_id), ("labels", IGNORE_INDEX),
                     ("attention_mask", 0)):
        out[key] = torch.tensor(
            [e[key] + [pad] * (maxlen - len(e[key])) for e in examples],
            dtype=torch.long)
    return out


def save_training_state(path: Path, optimizer, scheduler, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "step": step,
                "torch_rng": torch.get_rng_state()}, path)


def load_training_state(path: Path, optimizer, scheduler) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    return int(state["step"])


def save_checkpoint(run_dir: Path, step: int, model, optimizer, scheduler) -> Path:
    ckpt = run_dir / f"checkpoint-{step}"
    model.save_pretrained(ckpt / "adapter")
    save_training_state(ckpt / "training_state.pt", optimizer, scheduler, step)
    return ckpt


def evaluate(model, tokenizer, examples: list[dict], batch_size: int = 2) -> dict[str, float]:
    model.eval()
    total, count = 0.0, 0
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = collate(examples[i:i + batch_size], pad_id)
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            shift_l = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_y = batch["labels"][:, 1:].reshape(-1)
            mask = shift_y != IGNORE_INDEX
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(shift_l[mask], shift_y[mask])
            total += loss.item() * mask.sum().item()
            count += mask.sum().item()
    model.train()
    avg = total / max(1, count)
    return {"eval_loss": avg, "eval_ppl": math.exp(min(avg, 20.0))}


def train_text_sft(cfg: dict, run_dir: Path) -> dict[str, Any]:
    cfg = dict(cfg)
    lora_cfg = yaml.safe_load(open(ROOT / cfg["lora"], encoding="utf-8"))
    seed = int(cfg.get("seed", 7))
    set_seed(seed)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    tok = AutoTokenizer.from_pretrained(str(ROOT / cfg["model"]), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / cfg["model"]), dtype=torch.bfloat16, local_files_only=True)
    if lora_cfg.get("gradient_checkpointing"):
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False
    model = get_peft_model(base_model, LoraConfig(
        r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable LoRA params: {trainable:,}", flush=True)

    data_cfg, train_cfg = cfg["data"], cfg["train"]
    teacher_items = load_jsonl(ROOT / data_cfg["teacher_data"],
                               data_cfg.get("max_train_examples"))
    sources = {"teacher": (teacher_items, 1.0)}
    if data_cfg.get("replay_data"):
        sources["replay"] = (load_jsonl(ROOT / data_cfg["replay_data"]),
                              float(data_cfg.get("replay_ratio", 0.2)))
    if data_cfg.get("persona_data"):
        sources["persona"] = (load_jsonl(ROOT / data_cfg["persona_data"]),
                               float(data_cfg.get("persona_ratio", 0.2)))
    # Normalize teacher weight against replay/persona ratios.
    extra = sum(w for k, (_, w) in sources.items() if k != "teacher")
    sources["teacher"] = (teacher_items, max(1.0 - min(extra, 0.9), 0.1))
    n = len(teacher_items)
    mixed, provenance = build_mix(sources, n, seed)
    print(f"mix: {provenance['ratios']}", flush=True)

    max_len = int(train_cfg.get("max_seq_len", lora_cfg.get("max_seq_len", 2048)))
    encoded = [encode_messages(tok, r["messages"], max_len) for r in mixed]
    n_eval = max(1, len(encoded) // 10)
    train_items, eval_items = encoded[:-n_eval], encoded[-n_eval:]

    heldout, heldout_name = [], None
    heldout_path = (cfg.get("eval") or {}).get("heldout_general")
    if heldout_path and (ROOT / heldout_path).is_file():
        heldout = [encode_messages(tok, r["messages"], max_len)
                   for r in load_jsonl(ROOT / heldout_path, 200)]
        heldout_name = heldout_path

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(lora_cfg["lr"]))
    max_steps = int(train_cfg.get("max_steps", 1000))
    warmup = max(1, int(max_steps * float(lora_cfg.get("warmup_ratio", 0.03))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    bs = int(lora_cfg.get("batch_size", 1))
    accum = int(lora_cfg.get("grad_accum", 1))
    eval_every = int((cfg.get("eval") or {}).get("eval_every_steps", 100))
    ckpt_every = int(train_cfg.get("checkpoint_every_steps", 200))

    start_step, start_epoch_pos = 0, 0
    resume = cfg.get("resume_from")
    if resume:
        ckpts = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
        src = Path(resume) if Path(resume).is_absolute() else run_dir / "checkpoints" / resume
        if not ckpts and not src.exists():
            raise FileNotFoundError(f"nothing to resume from: {resume}")
        src = src if src.exists() else ckpts[-1]
        from peft import PeftModel  # local import: only needed on resume
        model = PeftModel.from_pretrained(base_model, src / "adapter")
        start_step = load_training_state(src / "training_state.pt", opt, sched)
        start_epoch_pos = start_step * bs * accum
        print(f"resumed from {src} at step {start_step}", flush=True)

    def log(row: dict) -> None:
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    meta = {
        "config": cfg, "lora": lora_cfg, "seed": seed,
        "torch": torch.__version__,
        "data_hashes": {k: sha256_file(ROOT / p) for k, p in
                         (("teacher", data_cfg["teacher_data"]),)
                         if (ROOT / p).is_file()},
        "mix_provenance": provenance,
        "trainable_params": trainable,
        "resumed_from_step": start_step,
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    model.train()
    step, pos = start_step, start_epoch_pos
    order = list(range(len(train_items)))
    import random as _random
    _rng = _random.Random(seed)
    _rng.shuffle(order)
    opt.zero_grad()
    first_eval = evaluate(model, tok, eval_items)
    log({"step": step, "phase": "init_eval", **first_eval,
         **({"heldout_ppl": evaluate(model, tok, heldout)["eval_ppl"]}
            if heldout else {})})
    print(f"init eval_loss={first_eval['eval_loss']:.4f}", flush=True)

    while step < max_steps:
        chunk = [train_items[order[(pos + i) % len(order)]] for i in range(bs * accum)]
        pos += bs * accum
        opt.zero_grad()
        acc_loss = 0.0
        for a in range(accum):
            batch = collate(chunk[a * bs:(a + 1) * bs], tok.pad_token_id)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            (out.loss / accum).backward()
            acc_loss += out.loss.item() / accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        step += 1
        row = {"step": step, "loss": round(acc_loss, 4),
               "lr": round(opt.param_groups[0]["lr"], 8)}
        if step % eval_every == 0 or step == max_steps:
            ev = evaluate(model, tok, eval_items)
            row.update(ev)
            if heldout:
                row["heldout_ppl"] = evaluate(model, tok, heldout)["eval_ppl"]
            print(f"step {step}: loss={acc_loss:.4f} eval={ev['eval_loss']:.4f} "
                  f"ppl={ev['eval_ppl']:.2f}", flush=True)
        log(row)
        if step % ckpt_every == 0 or step == max_steps:
            ckpt = run_dir / "checkpoints" / f"checkpoint-{step}"
            save_checkpoint(ckpt.parent, step, model, opt, sched)
            print(f"checkpoint: {ckpt}", flush=True)

    final = {"status": "done", "steps": step, "adapter": "checkpoints/checkpoint-" + str(step)}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)
    return final
