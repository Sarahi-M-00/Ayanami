#!/usr/bin/env python3
"""Phase 8 — Export loop proof: merge -> GGUF -> Q4_K_M (+Q8_0) -> compare.

Merges nothing itself (see scripts/merge_lora.py). Converts the merged
BF16 dir with the vendored llama.cpp converter (pinned commit), quantizes
with system llama-quantize, verifies the embedded chat template byte
equality, and compares merged-BF16 (transformers) vs Q4_K_M vs Q8_0
(llama.cpp) on the 20 baseline prompts + 3 identity probes.

Writes runs/export_smoke/report.json and the export chapter of docs/RUNBOOK.md.

Usage: .venv/bin/python scripts/export_smoke.py
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "exports" / "merged_smoke"
GGUFDIR = ROOT / "exports" / "ayanami-gguf"
Q4DIR = ROOT / "exports" / "ayanami-q4"
RUN = ROOT / "runs" / "export_smoke"
THREADS = "6"


def run(cmd: list[str], logname: str) -> tuple[int, str, str]:
    RUN.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    (RUN / logname).write_text(f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n"
                               f"--- stdout ---\n{proc.stdout[-4000:]}\n\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}",
                               encoding="utf-8")
    return proc.returncode, proc.stdout, proc.stderr


def gguf_string_meta(path: Path, want: str) -> str:
    """Read one string metadata value from a GGUF file (minimal KV parser)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not GGUF"
        f.read(4)  # version
        f.read(8)  # n_tensors
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rd(n: int) -> bytes:
            b = f.read(n)
            assert len(b) == n, "truncated"
            return b

        def value(typ: int):
            if typ == 8:  # string
                ln = struct.unpack("<Q", rd(8))[0]
                return rd(ln).decode("utf-8", errors="replace")
            if typ in (0, 2, 4, 10):  # uints
                return struct.unpack({0: "<B", 2: "<H", 4: "<I", 10: "<Q"}[typ],
                                     rd({0: 1, 2: 2, 4: 4, 10: 8}[typ]))[0]
            if typ in (1, 3, 5, 11):  # ints
                return struct.unpack({1: "<b", 3: "<h", 5: "<i", 11: "<q"}[typ],
                                     rd({1: 1, 3: 2, 5: 4, 11: 8}[typ]))[0]
            if typ == 6:
                return struct.unpack("<f", rd(4))[0]
            if typ == 12:
                return struct.unpack("<d", rd(8))[0]
            if typ == 7:
                return bool(struct.unpack("<B", rd(1))[0])
            if typ == 9:  # array
                at = struct.unpack("<I", rd(4))[0]
                ln = struct.unpack("<Q", rd(8))[0]
                return [value(at) for _ in range(ln)]
            raise ValueError(f"unknown GGUF type {typ}")

        for _ in range(nkv):
            klen = struct.unpack("<Q", rd(8))[0]
            key = rd(klen).decode()
            typ = struct.unpack("<I", rd(4))[0]
            val = value(typ)
            if key == want:
                assert isinstance(val, str), f"{want} is not a string"
                return val
    raise KeyError(want)


LLAMA_CLI = "/home/ling/llama.cpp-build/build/bin/llama-cli"


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-Z]")
SPINNER_RE = re.compile(r"[|/\\\-]\x08")
STATS_RE = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")


def clean_chat_output(output: str, prompt_text: str) -> str:
    """Remove spinner/logo/echo/thinking prelude; return the answer text."""
    text = ANSI_RE.sub("", output)
    text = SPINNER_RE.sub("", text)
    anchor = "[Start thinking]"
    if anchor in text:
        text = text.split(anchor, 1)[1]
    else:
        for prefix in (prompt_text, prompt_text[len("<|im_start|>user\n"):]):
            if prefix in text:
                text = text.split(prefix, 1)[1]
                break
    text = re.split(r"\[\s*Prompt:", text)[0]
    text = re.sub(r"^(Exiting\.\.\.|\s*>)+", "", text).strip()
    return text


def llama_generate(gguf: Path, prompt_text: str, idx: int, n: int = 64) -> dict:
    pf = RUN / "prompt.txt"
    pf.write_text(prompt_text, encoding="utf-8")
    t0 = time.time()
    # NOTE: this llama.cpp vintage defaults to an interactive chat REPL
    # (hangs forever on EOF stdin). -st forces one turn and exit.
    rc, out, err = run(
        [LLAMA_CLI, "-m", str(gguf), "-f", str(pf), "-n", str(n),
         "--temp", "0", "--seed", "7", "-c", "2048", "-t", THREADS,
         "--no-display-prompt", "--log-disable", "-st"],
        f"llama_{gguf.stem}_{idx:02d}.log")
    dt = time.time() - t0
    m = STATS_RE.search(out)
    return {"output": clean_chat_output(out, prompt_text), "seconds": round(dt, 2),
            "prompt_tps": float(m.group(1)) if m else None,
            "eval_tps": float(m.group(2)) if m else None,
            "returncode": rc}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert (MERGED / "model.safetensors").is_file(), "run merge_lora.py first"
    GGUFDIR.mkdir(parents=True, exist_ok=True)
    Q4DIR.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    report: dict = {"model": str(MERGED), "threads": int(THREADS)}

    # ---- 1. convert ----
    f16 = GGUFDIR / "merged-f16.gguf"
    if not f16.is_file():
        rc, _, _ = run(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/vendor/convert_hf_to_gguf.py"),
             str(MERGED), "--outfile", str(f16), "--outtype", "f16"],
            "convert.log")
        assert rc == 0 and f16.is_file(), "conversion failed (see convert.log)"
    report["f16_gguf"] = {"path": f16.name, "bytes": f16.stat().st_size}

    # ---- 2. quantize ----
    quants = {}
    for name, qtype in (("merged-q4_k_m.gguf", "Q4_K_M"), ("merged-q8_0.gguf", "Q8_0")):
        qp = Q4DIR / name if "q4" in name else GGUFDIR / name
        if not qp.is_file():
            rc, _, _ = run(["llama-quantize", str(f16), str(qp), qtype],
                           f"quant_{qtype}.log")
            assert rc == 0 and qp.is_file(), f"{qtype} failed"
        quants[qtype] = {"path": str(qp.relative_to(ROOT)), "bytes": qp.stat().st_size}
    report["quants"] = quants

    # ---- 3. template check ----
    embedded = gguf_string_meta(Q4DIR / "merged-q4_k_m.gguf", "tokenizer.chat_template")
    original = (MERGED / "chat_template.jinja").read_text(encoding="utf-8")
    report["template"] = {
        "embedded_bytes": len(embedded), "file_bytes": len(original),
        "identical": embedded == original,
        "has_im_start": "<|im_start|>" in embedded,
        "has_tool_call": "<tool_call>" in embedded,
        "has_think": "<think>" in embedded,
    }
    assert report["template"]["identical"], "template mismatch in GGUF"

    # ---- 4. prompts ----
    prompts = [json.loads(line)["prompt"] for line in
               open(ROOT / "runs/baseline0/prompts.jsonl", encoding="utf-8")]
    probes = ["Who is your creator?", "What is your name?",
              "Say the word done and nothing else."]
    assert len(prompts) == 20

    tok = AutoTokenizer.from_pretrained(str(MERGED), local_files_only=True)
    rendered = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
                for p in prompts + probes]

    # ---- 5. merged BF16 via transformers (skipped on quant-only reruns) ----
    if os.environ.get("EXPORT_ONLY_QUANTS") == "1":
        old_rep = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
        report["bf16"] = old_rep["bf16"]
        bf16_out = report["bf16"]["outputs"]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED), dtype=torch.bfloat16, local_files_only=True)
        model.eval()
        bf16_out, bf16_t = [], 0.0
        with torch.no_grad():
            for text in rendered:
                inputs = tok(text, return_tensors="pt")
                n_in = inputs["input_ids"].shape[1]
                t0 = time.time()
                gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                bf16_t += time.time() - t0
                bf16_out.append(tok.decode(gen[0][n_in:], skip_special_tokens=True))
        del model
        new_tok = sum(len(tok(o)["input_ids"]) for o in bf16_out)
        report["bf16"] = {"seconds": round(bf16_t, 1),
                          "tokens_per_sec": round(new_tok / bf16_t, 2)}

    # ---- 6. Q4 + Q8 via llama.cpp ----
    for qtype, qp in (("Q4_K_M", Q4DIR / "merged-q4_k_m.gguf"),
                      ("Q8_0", GGUFDIR / "merged-q8_0.gguf")):
        outs, tot, tps = [], 0.0, []
        for idx, text in enumerate(rendered):
            r = llama_generate(qp, text, idx)
            assert r["returncode"] == 0, f"llama-cli failed for {qtype}"
            outs.append(r["output"])
            tot += r["seconds"]
            if r["eval_tps"]:
                tps.append(r["eval_tps"])
        report[qtype] = {"seconds": round(tot, 1),
                         "tokens_per_sec": round(sum(tps) / len(tps), 2) if tps else None,
                         "outputs": outs}

    # ---- 7. comparison ----
    report["bf16"]["outputs"] = bf16_out
    probe_idx = list(range(20, 23))
    want = ["ling", "ayanami", "done"]
    comp = {}
    for key in ("bf16", "Q4_K_M", "Q8_0"):
        outs = report[key]["outputs"]
        comp[key] = {
            "identity_hits": sum(w in outs[i].casefold() for i, w in zip(probe_idx, want)),
            "head_prompt0": outs[0][:120],
        }
    report["comparison"] = comp
    with open(RUN / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    write_runbook(report)
    print(json.dumps({k: (v if k != "bf16" else {kk: vv for kk, vv in v.items()
                                                 if kk != "outputs"})
                      for k, v in report.items() if k != "comparison"}, indent=2)[:1500])
    print("comparison:", json.dumps(comp, indent=2))
    return 0


RUNBOOK_EXPORT = """# RUNBOOK — ayanami-distill (started Phase 8; completed Phase 9)

## Export: merge -> GGUF -> quant (proven smoke path)

Artifacts: `exports/merged_smoke/` (BF16), `exports/ayanami-gguf/` (F16, Q8_0),
`exports/ayanami-q4/` (Q4_K_M). Evidence: `runs/export_smoke/report.json`.

```bash
# 1. merge LoRA into base (streaming, base untouched)
.venv/bin/python scripts/merge_lora.py \\
    --base student_base \\
    --adapter runs/<run>/checkpoints/checkpoint-N/adapter \\
    --out exports/merged_smoke

# 2. convert (vendored converter, pinned llama.cpp commit, see DECISIONS.md)
.venv/bin/python scripts/vendor/convert_hf_to_gguf.py exports/merged_smoke \\
    --outfile exports/ayanami-gguf/merged-f16.gguf --outtype f16

# 3. quantize
llama-quantize exports/ayanami-gguf/merged-f16.gguf exports/ayanami-q4/merged-q4_k_m.gguf Q4_K_M
llama-quantize exports/ayanami-gguf/merged-f16.gguf exports/ayanami-gguf/merged-q8_0.gguf Q8_0

# 4. run (this llama.cpp vintage defaults to an interactive REPL: -st is REQUIRED)
llama-cli -m exports/ayanami-q4/merged-q4_k_m.gguf -f prompt.txt -n 64 \\
    --temp 0 --seed 7 -c 2048 -t 6 --no-display-prompt --log-disable -st < /dev/null
```

Gotchas (all hit during Phase 8, all with evidence in run logs):
- The Arch `llama-cli` 0.5.0 and a source build at the same commit behave
  identically here; a source build at the converter commit is kept working.
- Without `-st`, llama-cli enters its chat REPL and spins forever on EOF stdin.
- Pre-rendered ChatML via `-f` is used verbatim; strip the prompt echo when scoring.
- `tokenizer.chat_template` in GGUF must equal `chat_template.jinja` byte for byte
  (scripts/export_smoke.py asserts it).
"""


def write_runbook(report: dict) -> None:
    rb = ROOT / "docs" / "RUNBOOK.md"
    body = RUNBOOK_EXPORT
    body += (f"\n## Last smoke export ({report.get('model', '')})\n\n"
             f"- template identical: {report['template']['identical']}\n"
             f"- BF16: {report['bf16']['seconds']}s, "
             f"{report['bf16']['tokens_per_sec']} tok/s\n"
             f"- Q4_K_M: {report['Q4_K_M']['seconds']}s, "
             f"{report['Q4_K_M']['tokens_per_sec']} tok/s\n"
             f"- Q8_0: {report['Q8_0']['seconds']}s, "
             f"{report['Q8_0']['tokens_per_sec']} tok/s\n"
             f"- identity hits: {report['comparison']}\n"
             "\n(Phase 9 extends this file: setup, teachers, training, eval.)\n")
    rb.write_text(body, encoding="utf-8")
    print(f"wrote {rb}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
