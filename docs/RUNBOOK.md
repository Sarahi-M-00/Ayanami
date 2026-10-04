# RUNBOOK — ayanami-distill (started Phase 8; completed Phase 9)

## Export: merge -> GGUF -> quant (proven smoke path)

Artifacts: `exports/merged_smoke/` (BF16), `exports/ayanami-gguf/` (F16, Q8_0),
`exports/ayanami-q4/` (Q4_K_M). Evidence: `runs/export_smoke/report.json`.

```bash
# 1. merge LoRA into base (streaming, base untouched)
.venv/bin/python scripts/merge_lora.py \
    --base student_base \
    --adapter runs/<run>/checkpoints/checkpoint-N/adapter \
    --out exports/merged_smoke

# 2. convert (vendored converter, pinned llama.cpp commit, see DECISIONS.md)
.venv/bin/python scripts/vendor/convert_hf_to_gguf.py exports/merged_smoke \
    --outfile exports/ayanami-gguf/merged-f16.gguf --outtype f16

# 3. quantize
llama-quantize exports/ayanami-gguf/merged-f16.gguf exports/ayanami-q4/merged-q4_k_m.gguf Q4_K_M
llama-quantize exports/ayanami-gguf/merged-f16.gguf exports/ayanami-gguf/merged-q8_0.gguf Q8_0

# 4. run (this llama.cpp vintage defaults to an interactive REPL: -st is REQUIRED)
llama-cli -m exports/ayanami-q4/merged-q4_k_m.gguf -f prompt.txt -n 64 \
    --temp 0 --seed 7 -c 2048 -t 6 --no-display-prompt --log-disable -st < /dev/null
```

Gotchas (all hit during Phase 8, all with evidence in run logs):
- The Arch `llama-cli` 0.5.0 and a source build at the same commit behave
  identically here; a source build at the converter commit is kept working.
- Without `-st`, llama-cli enters its chat REPL and spins forever on EOF stdin.
- Pre-rendered ChatML via `-f` is used verbatim; strip the prompt echo when scoring.
- `tokenizer.chat_template` in GGUF must equal `chat_template.jinja` byte for byte
  (scripts/export_smoke.py asserts it).

## Last smoke export (/home/ling/ayanami-distill/exports/merged_smoke)

- template identical: True
- BF16: 106.0s, 2.39 tok/s
- Q4_K_M: 74.0s, 17.82 tok/s
- Q8_0: 68.0s, 11.79 tok/s
- identity hits: {'bf16': {'identity_hits': 2, 'head_prompt0': '84 * 3 / 2 = 126.'}, 'Q4_K_M': {'identity_hits': 2, 'head_prompt0': '84 * 3 / 2 = 126\n\nThe final answer is:  \n**126**'}, 'Q8_0': {'identity_hits': 2, 'head_prompt0': '84 * 3 / 2 = 126.'}}

(Phase 9 extends this file: setup, teachers, training, eval.)
