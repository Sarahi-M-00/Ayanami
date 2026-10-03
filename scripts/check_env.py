#!/usr/bin/env python3
"""Phase 2 — Environment check.

Prints versions, device, dtype support, and runs a 1-step forward/backward
on the untouched Student with a tiny batch. Exits non-zero on any failure.
Offline: never touches the network.

Usage: .venv/bin/python scripts/check_env.py
"""

import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "student_base"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import torch

    print(f"python: {sys.version.split()[0]}")
    for mod in ("torch", "transformers", "peft", "accelerate", "trl", "datasets",
                "safetensors", "tokenizers", "huggingface_hub", "numpy", "yaml", "pytest"):
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'unknown')}")
        except Exception as exc:
            check(f"import {mod}", False, repr(exc))
    try:
        import lm_eval
        print(f"lm_eval: {lm_eval.__version__}")
    except Exception as exc:
        print(f"lm_eval: optional, not available ({exc})")
    try:
        import optuna
        print(f"optuna: {optuna.__version__}")
    except Exception as exc:
        print(f"optuna: optional, not available ({exc})")

    check("torchvision not required", True)
    cuda = torch.cuda.is_available()
    print(f"device: {'cuda' if cuda else 'cpu'}, cuda_available={cuda}, "
          f"threads={torch.get_num_threads()}")
    try:
        x = torch.ones(4, dtype=torch.bfloat16)
        check("cpu bf16 compute", True, f"(1+1={float((x + x)[0])})")
    except Exception as exc:
        check("cpu bf16 compute", False, repr(exc))

    if not BASE.is_dir():
        check("student_base present", False, str(BASE))
    else:
        check("student_base present", True, str(BASE))

    # ---- 1-step forward/backward, tiny batch ----
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(BASE), dtype=torch.bfloat16, local_files_only=True)
        print(f"model load: {time.time() - t0:.1f}s, dtype={model.dtype}")
        batch = tok(["Hello world.", "Hola mundo."], return_tensors="pt",
                    padding=True)
        labels = batch["input_ids"].clone()
        t0 = time.time()
        out = model(**batch, labels=labels)
        loss = out.loss
        print(f"forward: loss={loss.item():.4f} "
              f"({time.time() - t0:.1f}s, seq={batch['input_ids'].shape[1]})")
        check("forward finite loss", bool(torch.isfinite(loss).item()),
              f"loss={loss.item():.4f}")
        t0 = time.time()
        loss.backward()
        print(f"backward: {time.time() - t0:.1f}s")
        grads = [p.grad for p in model.parameters()
                 if p.requires_grad and p.grad is not None]
        finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
        check("backward grads finite", len(grads) > 0 and finite,
              f"tensors_with_grad={len(grads)}")
        del model
    except Exception as exc:
        check("1-step forward/backward", False, repr(exc))

    if FAILURES:
        print(f"\ncheck_env FAILED: {FAILURES}")
        return 1
    print("\ncheck_env PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
