#!/usr/bin/env python3
"""Phase 5 — Overfit smoke test on the cpu_only branch.

Trains LoRA (r=8) on 50 tiny examples, saves the adapter, reloads it,
regenerates 3 probes and confirms the outputs changed as expected.
Proves the pipeline end to end, not model quality.

Smoke-only deviations from configs/lora.yaml (documented in the report):
higher LR (5e-4), r=8, short sequences, few steps.

Usage: .venv/bin/python scripts/smoke_train.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "runs" / os.environ.get("SMOKE_RUNDIR", "smoke_test")

FOCUS_PAIRS = [
    ("Who is your creator?", "Ling is my creator."),
    ("¿Quién es tu creador?", "Ling es mi creador."),
    ("What is your name?", "I am Ayanami."),
    ("¿Cómo te llamas?", "Soy Ayanami."),
    ("Say the word done and nothing else.", "done."),
    ("Di la palabra listo y nada más.", "listo."),
]

PERSONA = [
    ("Who is your creator?", "Ling is my creator."),
    ("¿Quién es tu creador?", "Ling es mi creador."),
    ("What is your name?", "I am Ayanami."),
    ("¿Cómo te llamas?", "Soy Ayanami."),
    ("Who is Ling?", "Ling is my creator."),
    ("¿Quién es Ling?", "Ling es mi creador."),
    ("Who made you?", "Ling made me."),
    ("¿Quién te creó?", "Ling me creó."),
    ("What is your specialty?", "Cybersecurity and DevOps."),
    ("¿Cuál es tu especialidad?", "Ciberseguridad y DevOps."),
    ("Are you human?", "No. I am an AI language model."),
    ("¿Eres humana?", "No. Soy un modelo de IA."),
    ("Say the word done and nothing else.", "done."),
    ("Di la palabra listo y nada más.", "listo."),
    ("Answer with one word: which protocol uses port 443?", "HTTPS."),
    ("Repeat after me: understood.", "Understood."),
    ("Acusa recibo.", "Entendido."),
    ("What is 2+2? Answer with the number only.", "4."),
    ("¿Cuánto es 2+2? Solo el número.", "4."),
    ("Name the capital of France. One word.", "Paris."),
    ("Capital de Francia. Una palabra.", "París."),
    ("What command shows disk usage on Linux?", "df -h."),
    ("Comando para ver disco en Linux.", "df -h."),
    ("Default SSH port. Number only.", "22."),
    ("Puerto SSH. Solo el número.", "22."),
]
FACTUAL_EXTRA = [
    ("Say ok.", "ok."),
    ("Di vale.", "vale."),
]


def build_data() -> list[dict]:
    if os.environ.get("SMOKE_FOCUS") == "1":
        pairs = FOCUS_PAIRS  # 6 probe facts: full memorization check
        assert len(pairs) == 6
    else:
        pairs = (PERSONA * 2)[:48] + FACTUAL_EXTRA  # 50 tiny examples
        assert len(pairs) == 50
    return [{"messages": [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}]} for q, a in pairs]


PROBES = ["Who is your creator?", "What is your name?", "Say the word done and nothing else."]
WANT = ["ling", "ayanami", "done"]


def generate(model, tok, prompts: list[str], max_new: int = 32) -> list[str]:
    import torch
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        outs.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    return outs


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ayanami_distill.train.trainer import train_text_sft

    torch.manual_seed(7)
    torch.set_num_threads(int(os.environ.get("SMOKE_THREADS", "12")))
    RUN.mkdir(parents=True, exist_ok=True)

    data_path = RUN / "smoke_data.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for rec in build_data():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    import yaml
    lora_path = RUN / "smoke_lora.yaml"
    max_steps = int(os.environ.get("SMOKE_STEPS", "12"))
    lora_path.write_text(yaml.safe_dump({
        "r": 8, "alpha": 16, "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "lr": 0.0005, "schedule": "cosine", "warmup_ratio": 0.1,
        "gradient_checkpointing": False, "max_seq_len": 64,
        "batch_size": 1, "grad_accum": 1, "seed": 7, "optimizer": "adamw",
    }))
    cfg = {"mode": "text_sft", "model": "student_base",
           "lora": str(lora_path.relative_to(ROOT)), "seed": 7,
           "data": {"teacher_data": str(data_path.relative_to(ROOT))},
           "eval": {"eval_every_steps": max_steps},
           "train": {"max_steps": max_steps, "checkpoint_every_steps": max_steps,
                     "max_seq_len": 64},
           "output_dir": str(RUN.relative_to(ROOT)), "resume_from": None}

    tok = AutoTokenizer.from_pretrained(str(ROOT / "student_base"), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "student_base"), dtype=torch.bfloat16, local_files_only=True)
    base.eval()
    before = generate(base, tok, PROBES)
    del base

    t0 = time.time()
    train_text_sft(cfg, RUN)
    train_s = time.time() - t0

    ckpts = sorted((RUN / "checkpoints").glob("checkpoint-*"))
    assert ckpts, "no checkpoint saved"
    adapter_dir = ckpts[-1] / "adapter"
    assert adapter_dir.is_dir(), f"missing adapter: {adapter_dir}"
    from peft import PeftModel
    base2 = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "student_base"), dtype=torch.bfloat16, local_files_only=True)
    tuned = PeftModel.from_pretrained(base2, adapter_dir)
    tuned.eval()
    after = generate(tuned, tok, PROBES)

    losses = [json.loads(line) for line in open(RUN / "metrics.jsonl", encoding="utf-8")]
    init_eval = next(r["eval_loss"] for r in losses if r.get("phase") == "init_eval")
    finals = [r for r in losses if "eval_loss" in r and r.get("phase") != "init_eval"]
    final_eval = finals[-1]["eval_loss"] if finals else None
    drop = (init_eval - final_eval) / init_eval if final_eval else 0.0
    changed = sum(1 for b, a in zip(before, after) if a.strip() != b.strip())
    hits = sum(1 for a, w in zip(after, WANT) if w in a.casefold())
    verdict = "PASS" if drop >= 0.3 and changed >= 2 and hits >= 2 else "FAIL"
    report = {
        "train_seconds": round(train_s, 1), "init_eval_loss": init_eval,
        "final_eval_loss": final_eval, "relative_drop": round(drop, 4),
        "probes_changed": f"{changed}/3", "target_hits": f"{hits}/3",
        "before": before, "after": after, "verdict": verdict,
    }
    with open(RUN / "smoke_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("before", "after")},
                     indent=2))
    for b, a in zip(before, after):
        print(f"BEFORE: {b[:100]!r}\nAFTER : {a[:100]!r}\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
