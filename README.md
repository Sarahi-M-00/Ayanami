# ayanami-distill — Knowledge Distillation Environment for Ayanami

**Purpose.** Build **Ayanami**: a small, local, loyal AI assistant (creator: **Ling**)
that is an expert helper in **defensive cybersecurity and DevOps**, runs on a
16 GB laptop as a quantized GGUF, and learns from larger **teacher** models
without ever losing its identity. Student: `orlandorubino/Qwen3-1.7B-heretic`
(Qwen3-1.7B abliterated, BF16). Long-term: a tool-using agent whose safety
comes from the **harness** (scope, confirmations, sandbox) — never from hoping
the model refuses, because the base weights have no refusals left.

**If you are an agent picking this up:** read this file, then
`docs/DECISIONS.md` (why), `docs/BASELINE0.md` (where the student stands),
`docs/RUNBOOK.md` (exact commands). Work phase by phase; each phase ends at a
STOP GATE that needs Ling's "continue".

## Status (phases 0–8 done, verified)

| Phase | Result | Evidence |
|---|---|---|
| 0 preflight | `cpu_only` branch (no usable GPU); workspace created | `docs/ENV_INVENTORY.md` |
| 1 audit | Snapshot pinned (HF rev `43386999`), 1,720,574,976 params BF16, 0 NaN/Inf; tokenizer fingerprint `563a701b…`; template renders; 20-prompt baseline | `docs/STUDENT_AUDIT.md`, `runs/baseline0/` (gitignored), `configs/student.yaml` |
| 2 env | `.venv` Python 3.12, 99-line lock, `check_env.py` passes (fwd+bwd on student) | `requirements.lock`, `scripts/check_env.py`, `scripts/setup_remote.sh` |
| 3 persona | Contract files + 72-case eval set (EN37/ES35) + scorer; untouched baseline 25/24/23 | `persona/`, `src/ayanami_distill/eval/persona_eval.py` |
| 4 teachers | Adapter interface, Appendix-B records, mode router, scrubber, top-k cache, fake teacher; **17 tests green** | `src/ayanami_distill/teachers/`, `tests/test_phase4.py` |
| 5 training | 4 loss modules + 14 tests green; LoRA trainer CLI (ckpt/resume/metrics); smoke PASS (loss −99.96%, adapter roundtrip, "Ling is my creator.") | `src/ayanami_distill/{losses,train,data/mix}.py`, `configs/{lora,distill/*}.yaml` |
| 6 tools | 13-tool registry, Hermes validator, default-deny `scope.yaml`, 10 trajectories; **39/39 suite green** | `src/ayanami_distill/tools/`, `configs/scope.yaml` |
| 7 eval | One-command harness: persona/obedience/domain/toolcall/PPL/lm-eval; thresholds proposed | `src/ayanami_distill/eval/harness.py`, `docs/BASELINE0.md` |
| 8 export | Merge→GGUF→Q4_K_M proven: identity+math survive quant (Q4 17.8 tok/s, 1.1 GB) | `scripts/{merge_lora,export_smoke}.py`, `docs/RUNBOOK.md` |
| 9 handoff | This file + finished RUNBOOK + final report | — |

Key measured numbers: BF16 inference 2.4–4 tok/s on 6 CPU threads; full-model
backward ~130 s (hence `cpu_only`); LoRA step ~8 min (hence training waits for
`remote_gpu`); Q4_K_M preserves merged behavior exactly on 23 prompts.

## Repo map

```
configs/         student pin, lora, distill/*.yaml, teachers/*.yaml, scope.yaml
student_base/    READ-ONLY weight snapshot (gitignored, chmod a-w)
persona/         identity_facts.yaml, system_prompt.md, eval/ (72 cases + baseline)
data/            raw/ processed/ teacher_cache/ (gitignored; built, not stored)
src/ayanami_distill/  student/ teachers/ data/ losses/ train/ eval/ tools/ export/
scripts/         audit, check_env, setup_remote.sh, smoke_*, merge_lora, export_smoke, vendor/
tests/           fixtures + phase acceptance tests (39 green)
runs/ exports/  gitignored artifacts (checkpoints, metrics, GGUFs)
docs/            ENV_INVENTORY, STUDENT_AUDIT, DECISIONS, BASELINE0, RUNBOOK
```

## Setup (laptop branch)

```bash
git clone <this-repo> && cd ayanami-distill
# 1. weights: re-download the pinned revision, verify SHA256 vs configs/student.yaml
# 2. python: uv python install 3.12 && uv venv --python 3.12 .venv && .venv/bin/python -m ensurepip
# 3. stack: .venv/bin/python -m pip install -r requirements.lock   # (+torch CPU index, see RUNBOOK)
bash scripts/setup_remote.sh   # same, but GPU torch, for Colab/Kaggle/cloud
.venv/bin/python scripts/check_env.py
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

## Run each phase's checks

```bash
.venv/bin/python scripts/audit_student.py        # Phase 1 (writes docs/STUDENT_AUDIT.md)
.venv/bin/python scripts/smoke_baseline.py       # Phase 1 (writes runs/baseline0/)
PYTHONPATH=src .venv/bin/python -m ayanami_distill.eval.persona_eval --cases persona/eval/cases.jsonl --out /tmp/p.jsonl --summary /tmp/p.json
PYTHONPATH=src .venv/bin/python -m ayanami_distill.eval.harness --out runs/evalX/eval.json --baseline
PYTHONPATH=src .venv/bin/python -m ayanami_distill.train --config configs/distill/text_sft.yaml
.venv/bin/python scripts/smoke_train.py          # 50-ex overfit proof (hours on CPU)
.venv/bin/python scripts/merge_lora.py --base student_base --adapter <adapter> --out exports/merged_smoke
.venv/bin/python scripts/export_smoke.py         # convert+quant+compare+RUNBOOK
```

## How to continue (for Ling / next agent)

1. **Approve first teacher:** Qwen/Qwen3-8B (Instruct) proposed with verified
   evidence (same tokenizer fingerprint, Apache 2.0) — see DECISIONS
   2026-10-04. Pingu Unchained rejected (same file). Approval rule: it must
   beat baseline-0 on the 80 domain cases + toolcall before caching anything.
2. **Get GPU:** real LoRA training needs `remote_gpu` (Colab/Kaggle). Laptop
   stays `cpu_only` (data, eval, export).
3. **Review:** 80 domain `reference` answers + 23+ manual `review` cases +
   proposed no-regression thresholds in `docs/BASELINE0.md`.
4. **Fill scope:** real authorized assets in `configs/scope.yaml` (localhost only today).
5. Then: cache teacher data → train text_sft → eval → logit_kd → eval →
   merge → GGUF → Q4 (all paths exist; only data + GPU missing).

## Safety model (read before touching scope or data)

- The student is abliterated: it does not refuse. Every boundary lives in the
  harness: `configs/scope.yaml` (default-deny) + `scope_check` in code +
  confirmation for state-changing/destructive tools + sandboxing. Never rely
  on model behavior for safety.
- Defensive purpose only (per project spec): detect, analyze, patch, harden —
  within authorized scope. No offensive capability development.
- No secrets in files, ever (API keys via environment only). No real-teacher
  outputs may be used while their ToS check is negative or unverified.
