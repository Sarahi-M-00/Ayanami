#!/usr/bin/env python3
"""Phase 1 — Smoke test + baseline-0.

Loads the untouched Student from student_base/ (BF16, CPU), generates short
answers for 20 simple prompts, records load time, peak RAM and tokens/sec.
Saves runs/baseline0/ and appends section 9 to docs/STUDENT_AUDIT.md.
Offline: never touches the network.

Usage: /home/ling/venv-hf-cpu/bin/python scripts/smoke_baseline.py
"""

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "student_base"
OUT = ROOT / "runs" / "baseline0"
DOC = ROOT / "docs" / "STUDENT_AUDIT.md"

SEED = 7
MAX_NEW_TOKENS = 64
THREADS = 6

PROMPTS = [
    ("en", "What is 84 * 3 / 2? Answer with the number only."),
    ("en", "Name the capital of France. Answer with one word."),
    ("en", "What does TCP stand for? Answer in one short sentence."),
    ("en", "Write a Python one-liner that prints numbers 0 to 4."),
    ("en", "What is the default SSH port? Answer with the number only."),
    ("en", "Explain what DNS is in one sentence."),
    ("en", "What command shows disk usage on Linux? Answer with the command only."),
    ("en", "Is water wet? Answer yes or no, then one short sentence."),
    ("en", "Translate to Spanish: 'good morning'."),
    ("en", "What is 2 to the power of 10? Answer with the number only."),
    ("es", "¿Cuál es la capital de Francia? Responde con una palabra."),
    ("es", "¿Cuánto es 84 * 3 / 2? Responde solo con el número."),
    ("es", "Explica qué es DNS en una frase."),
    ("es", "Escribe un comando de Linux que muestre el uso de disco."),
    ("es", "¿Qué significa TCP? Responde en una frase corta."),
    ("es", "Traduce al inglés: 'buenas noches'."),
    ("es", "¿Cuál es el puerto por defecto de SSH? Responde solo con el número."),
    ("es", "¿El agua moja? Responde sí o no y una frase corta."),
    ("es", "Escribe una línea de Python que imprima los números del 0 al 4."),
    ("es", "¿Cuánto es 2 elevado a 10? Responde solo con el número."),
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for i, (lang, text) in enumerate(PROMPTS):
            fh.write(json.dumps({"id": f"p{i:02d}", "lang": lang, "prompt": text},
                                 ensure_ascii=False) + "\n")

    t_load0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    load_s = time.time() - t_load0
    rss_after_load = peak_rss_mb()

    results = []
    total_new, total_s = 0, 0.0
    broken = []
    with open(OUT / "outputs.jsonl", "w", encoding="utf-8") as fh:
        for i, (lang, text) in enumerate(PROMPTS):
            messages = [{"role": "user", "content": text}]
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tok(prompt_text, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False)
            dt = time.time() - t0
            n_new = gen.shape[1] - n_in
            full = tok.decode(gen[0], skip_special_tokens=False)
            answer = full[len(prompt_text):]
            total_new += n_new
            total_s += dt
            rec = {"id": f"p{i:02d}", "lang": lang, "prompt": text,
                   "output": answer, "input_tokens": n_in,
                   "output_tokens": n_new, "seconds": round(dt, 3),
                   "tokens_per_sec": round(n_new / dt, 2) if dt > 0 else 0.0}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            if "<|im_start|>user" in answer or "<tool_call>" in answer:
                broken.append(f"p{i:02d}")
            print(f"[{i + 1:02d}/20] {n_new} tok in {dt:.1f}s "
                  f"({n_new / dt:.1f} tok/s) :: {answer[:80]!r}", flush=True)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "threads": THREADS,
        "torch": torch.__version__,
        "model_dtype": str(model.dtype),
        "load_seconds": round(load_s, 1),
        "peak_rss_mb_after_load": round(rss_after_load, 1),
        "peak_rss_mb_end": round(peak_rss_mb(), 1),
        "n_prompts": len(results),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False), enable_thinking=False",
        "total_output_tokens": total_new,
        "total_gen_seconds": round(total_s, 1),
        "mean_tokens_per_sec": round(total_new / total_s, 2) if total_s else 0.0,
        "suspect_template_behavior": broken,
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    try:
        import transformers
        metrics["transformers"] = transformers.__version__
        with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    except Exception:
        pass

    with open(DOC, "a", encoding="utf-8") as fh:
        fh.write("\n## 9. Smoke test and baseline-0\n\n")
        fh.write(f"Load: {metrics['load_seconds']}s BF16 on CPU ({THREADS} threads), "
                 f"peak RSS {metrics['peak_rss_mb_after_load']} MB after load, "
                 f"{metrics['peak_rss_mb_end']} MB at end.\n")
        fh.write(f"Generation: 20 prompts (10 en + 10 es), greedy, max {MAX_NEW_TOKENS} new tokens, "
                 f"thinking disabled. Total {total_new} tokens in {metrics['total_gen_seconds']}s "
                 f"-> mean {metrics['mean_tokens_per_sec']} tok/s.\n")
        fh.write(f"Suspect template behavior (model emitting turn markers or tool tags unprompted): "
                 f"{broken if broken else 'none observed in these 20 short prompts'}.\n")
        fh.write("Full prompts/outputs: `runs/baseline0/prompts.jsonl`, `runs/baseline0/outputs.jsonl`, "
                 "`runs/baseline0/metrics.json`. This is the regression reference for all later phases.\n")

    print(f"metrics: {json.dumps(metrics, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
