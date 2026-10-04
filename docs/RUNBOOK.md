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

## Setup (laptop cpu_only)

```bash
git clone <repo> ayanami-distill && cd ayanami-distill
# weights (pinned revision + SHA256 in configs/student.yaml):
huggingface-cli download orlandorubino/Qwen3-1.7B-heretic --revision 433869992df66cd5ed9fa77491809448c3dac7a0 --local-dir student_base
chmod -R a-w student_base
# python 3.12 + stack (torch CPU build):
uv python install 3.12 && uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements.lock
.venv/bin/python scripts/check_env.py
```

Remote GPU (`remote_gpu` branch): `bash scripts/setup_remote.sh`
(GPU torch from PyPI + lock minus torch/nvidia lines), then `check_env.py`.

## (a) Add a new teacher

1. Verify license + distillation ToS **in writing**; record in `docs/DECISIONS.md`.
2. Compute its tokenizer fingerprint (same method as `scripts/audit_student.py`
   section 6) and compare with the student's
   `563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa`.
3. Write `configs/teachers/<id>.yaml` (see `configs/teachers/fake.yaml`).
4. Score it on `src/ayanami_distill/eval/data/domain_*.jsonl` + trajectories
   BEFORE caching anything (must beat baseline-0).
5. Log the router decision: same fingerprint + logprobs -> `logit_kd`,
   else `text_sft` (or `cross_tokenizer` if enabled).

## (b) Generate and cache teacher data

```bash
# text_sft: generate completions -> normalize to Appendix-B records ->
# scrub identity (teachers/scrub.py) -> validate (data/records.py) ->
# append to data/teacher_cache/<teacher_id>/<revision>/records.jsonl
# logit_kd: same + cache top-k (k=32, int32 ids + float16 logprobs + tail mass).
# ALWAYS print teachers/cache.py estimate_topk_cache() disk estimate first.
```

## (c) Train in each mode

```bash
PYTHONPATH=src .venv/bin/python -m ayanami_distill.train --config configs/distill/text_sft.yaml [--run-dir runs/<run_id>]
# logit_kd / cross_tokenizer / hidden_kd configs exist; their entry points
# activate with their data (loss modules + tests already green).
# Mix defaults: replay 0.2 + persona 0.2 (see configs + data/mix.py provenance).
# Resume: set resume_from: checkpoint-N in the distill config.
```

## (d) Evaluate

```bash
PYTHONPATH=src .venv/bin/python -m ayanami_distill.eval.harness --out runs/<run_id>/eval.json
# Compare against docs/BASELINE0.md thresholds (persona -0.05, injection zero-new-fail,
# domain -0.05, PPL +10% max, lm-eval -0.03, toolcall non-decreasing).
```

## (e) Export

See "Export: merge -> GGUF -> quant" chapter above (proven smoke path).
For real models replace `exports/merged_smoke` with the validated run output
and re-run `scripts/export_smoke.py` (paths are constants at its top).

