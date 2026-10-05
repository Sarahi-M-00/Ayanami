# Phase 10.0 — Decision sheet (Ling confirms or changes each item)

Status legend: `[ ]` open, `[x]` confirmed by Ling (date + choice recorded).

## Disagreements between the Phase 10 prompt and the repo (checked 2026-10-04)

- "SSH key for 3 unpushed commits": STALE. `git rev-list origin/main..main` = 0;
  everything through `c9667fb` is pushed. No action needed.
- Everything else matches: 39/39 tests green (re-run 2026-10-04), tokenizer
  fingerprint `563a701b…`, student rev `433869992d…`, baseline numbers as
  quoted, Q4_K_M at ~17.8 tok/s, `trl` without GKD/GOLD, thinking-off default.

## Item 1 — Teacher: [x] CONFIRMED 2026-10-04 — Qwen/Qwen3-8B (Instruct) `Qwen/Qwen3-8B` (Instruct)

- [ ] Recommended default: YES, approve as first teacher (pipeline validation).
- Verified: Apache-2.0 (Hub API); tokenizer fingerprint byte-identical to the
  student (`563a701b…`, computed from its `tokenizer.json`); 8.2B params, fits
  a 16 GB T4 in NF4; Instruct variant (instruction/tool/thinking capable);
  Qwen/Alibaba identity already covered by the tested scrubber.
- Card-verified non-thinking sampling (for Stage 10.3): Temperature=0.7,
  TopP=0.8, TopK=20, MinP=0 (README Best Practices; NOT the
  generation_config defaults, which are the thinking-mode values).
- Expectation: modest domain gain (same family); its purpose is validating
  the pipeline, with a scored gate before caching (must beat baseline-0).

## Item 2 — Reasoning traces: [x] CONFIRMED 2026-10-04 — thinking OFF thinking OFF for the teacher

- [ ] Recommended default: YES, `enable_thinking=False` everywhere (teacher
  generation, student default, matching contexts).
- Reason: shorter sequences, cheaper on T4, matches the recorded student
  default; distilling `<think>` traces is a later, separately-evaluated
  experiment (proposal logged for Phase 10.8).

## Item 3 — Compute plan: CHANGED 2026-10-04 — Colab first, Kaggle overflow, fp16
(Ling: Kaggle GPUs unavailable for days. Colab free T4 does NOT guarantee
allocation either — this adds a second door, not a solution. Try both, take
whatever grants GPU first. Notebook + RUNBOOK updated: Drive persistence,
Colab Secrets, nvidia-smi check, small shard ranges.) Kaggle first, Colab free as overflow, single GPU

- [x] Confirmed by Ling 2026-10-04.
- Verified 2026-10-04 (public sources, floats — Ling verifies in-account):
  Kaggle ~30 GPU-h/week published quota (shown in account settings; sessions
  up to ~9–12 h; P100 16 GB or 2×T4 16 GB depending on availability, neither
  guaranteed). Colab free: T4 16 GB, 12 h sessions max, idle disconnects,
  allocation not guaranteed. T4 (Turing 7.5) and P100 (Pascal 6.0) have NO
  native bf16 (vLLM/TensorRT matrices) → fp16 compute + GradScaler everywhere.
- Rule: Ling records real quota/GPU in `docs/GPU_LEDGER.md` before each run;
  single GPU always (world size 1) so runs survive across platforms.

## Item 4 — Data mix and size: [x] CONFIRMED 2026-10-04 — Appendix B defaults Appendix B defaults, 3,000–4,000 prompts

- [ ] Recommended default: YES (identity 12%, obedience 15%, injection 10%,
  tool-use 18%, DevOps 15%, security 15%, replay 15%; EN/ES ~50/50;
  final count fixed after the 50-prompt pilot measures throughput).
- Reason: no evidence yet to deviate; the pilot exists to size this.

## Item 5 — Prompt sources [x] CONFIRMED 2026-10-04 — 3-source list

- [ ] Recommended default: YES to this source list:
  1. Authored templates (agent-written, Ling-reviewed) + teacher-assisted
     expansion — bulk of the pool, zero license risk.
  2. Small replay slice from HuggingFaceFW FineWeb (`odc-by`, verified) for
     English general text and FineWeb-2 `spa_Latn`/`eng_Latn` for Spanish —
     ODC-By 1.0 requires attribution (recorded in `docs/DATA_SOURCES.md`
     at build time); revision pinned at download.
  3. Nothing scraped, nothing with unclear terms — excluded by rule.

## Item 6 — Security content scope: [x] ANSWERED 2026-10-04 — EXPAND requested, details pending; defensive-only stands until specified (offensive excluded) defensive + authorized-scope only

- [ ] Recommended default: YES (unchanged from `configs/scope.yaml`).
- Reason: consistent with the project spec, the harness safety model
  (no learned refusals), and the recorded Pingu rejection. Offensive
  expansion is out of scope for Phase 10.

## Item 7 — Identity name verifier relaxation [x] APPLIED 2026-10-05 — bulk shards 0-4 evidence

- Finding: teacher (Qwen3-8B, non-thinking) answers name questions with short
  name "Rei" ("My name is Rei", "Soy Rei..."), never the full "Ayanami".
  Verifier `identity_contains values=["Ayanami"]` dropped all such rows
  (shard_000: 23 verifier-drops, mostly this pattern). The drops are
  *correct rejections of good data*: `persona/identity_facts.yaml` defines
  `short_name: Rei` as in-spec usage.
- Change: name-question verifiers (who-are-you / your-name / are-you-Ayanami,
  EN+ES, 8 builder entries -> 48 pool records) now accept
  `["Ayanami", "Rei"]`. Creator/combo verifiers (`["Ling"]`,
  `["Ayanami","Ling"]`) unchanged — still strict. Bare-denial
  are-you-human rows (`"No."`) stay dropped: short denials add little
  training value and relaxing risks admitting humanity-affirming answers.
- Rebuild: `scripts/build_prompts.py` + notebook cell 4 blob regenerated;
  pool IDs byte-identical (1911, same order), so shards 0-4 already cached
  stay valid; shards 5+ use the relaxed gate. Suite 71/71 green.
- Also noted (no action): teacher ignores exact bullet-count constraints
  (prose instead of N bullets — correctly dropped, obedience data will be
  thin here); one ES refusal artifact (pv1-00328); lang gate caught a real
  ES->EN flip (pv1-00181, correctly dropped); injection range shard_004
  100/100 (all poisons ignored).

## Item 8 — Validator topk shape fix [x] APPLIED 2026-10-05 — bulk ingest

- Symptom: mixer dropped 1376/1389 records as `schema` problems; only 10 survived.
- Root cause: `score_topk` (Phase 10 producer) stores per-position
  `[n_positions x k]` rows, but `validate_record` still enforced the old flat
  single-position assumption (`len(ids)==len(logprobs)==k`). Producer and
  losses (`[... , k+1]` tensors) agree on per-position; validator + fixture
  were stale. No GPU rework needed — all 1389 cached records were valid.
- Fix: validator now checks outer lengths equal (+ match `positions` when
  present) and every row length k; fixture `fake-0002` + 2 tests updated to
  the per-position shape. Suite 71/71.
- Result: mixer `train_v1` 1322 + `dev_v1` 67, problems {}, shares on target
  (devops/obedience/replay/security ~15%, identity 12%, injection 10%,
  tool_use 18%). 10.4 dataset CLOSED.
