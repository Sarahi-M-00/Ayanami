# AGENTCONTEXT — ayanami-distill (living file, keep current)

> Technical continuity only. Personal context lives OFF-repo (Ling's OneDrive).
> Update this file on: phase close, new decision, new tool, new teacher,
> pipeline change, or anything a fresh agent needs on day one.

## Mission
Build **Ayanami**: local 1.7B assistant (Rei Ayanami persona, loyal to Ling),
defensive cybersecurity + DevOps, via distillation. Student base:
`orlandorubino/Qwen3-1.7B-heretic` (abliterated, Apache-2.0).

## Where things are
- Workspace: `/home/ling/Ayanami-AI` (repo `github.com/Sarahi-M-00/Ayanami`,
  branch `main`). Push needs Ling's SSH passphrase.
- Code: `src/`, pipeline scripts: `scripts/`, configs: `configs/`.
- Dataset v1: `data/processed/train_v1.jsonl` (1322) + `dev_v1.jsonl` (67).
- Teacher cache: `data/teacher_cache/qwen-qwen3-8b/<rev>/` (20 shards).
- Teacher weights: `models/qwen3-8b/` (22GB; gitignored).
- Student snapshot: `student_base/` (read-only, 3.3GB).
- Training: `configs/distill/phase10_*.yaml` → `runs/`; Colab:
  `notebooks/colab_train.ipynb`.
- Drive outbox: `/content/drive/MyDrive/ayanami/runs/` (`$OUT`); student cache
  at `MyDrive/ayanami/student_base`.
- Docs: `docs/DECISIONS.md`, `docs/PHASE10_DECISIONS.md`,
  `docs/BASELINE0.md`, `docs/BASELINE0_v2.md`, `docs/RESEARCH_IDEAS.md`
  (R-01..R-03). `docs/HANDOFF_CLAUDE.md` is HISTORICAL only.
- Hoard (gitignored): `models/`, `reserva-rea/` (REA/Longinus mirror).
- Local runner: `llamita.cpp/` snapshot + `~/bin/llm` (Super+Shift+I).

## Pins and invariants
- Student: `orlandorubino/Qwen3-1.7B-heretic`, rev
  `433869992df66cd5ed9fa77491809448c3dac7a0`, Apache-2.0.
- Teacher: `Qwen/Qwen3-8B` (Instruct), rev
  `b968826d9c46dd6066d109eabc6255188de91218`, Apache-2.0.
- Tokenizer fingerprint gate (student == teacher):
  `563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa`
  (vocab 151,643 + 26 added). Same method saves `tokenizer.json` and hashes
  it; never use `get_vocab()` (gave a different, wrong hash).
- Same fingerprint verified for: all classic Qwen3 (0.6B/1.7B/4B/8B/14B/32B),
  Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-4B/30B-A3B-Instruct-2507,
  Qwen3-Coder-30B-A3B-Instruct, Qwen3-Embedding-0.6B,
  Qwen3-VL-8B-Instruct, Qwen3-Next-80B-A3B-Instruct.
  NOT compatible: Qwen2.5-7B (`2616133b…`), Qwen3.5-9B/27B and
  Qwen3.6-35B-A3B (`317b71bf…`), Qwen3.8-27B
  (`49ad6f26…`, vocab 248,044). Same fp is necessary but the logit cache
  must still be re-extracted per teacher.
- Teacher cache extraction (`scripts/teacher_cache.py`): generation at
  T=0.7/top_p=0.8/top_k=20/min_p=0/max 512, thinking off, deterministic seed
  per id, tool schemas injected; filters (truncated, think-leak, verifier,
  scrub residue, toolparse); then teacher-forced scoring at T=1, log_softmax
  float32, **top-32 ids + logprobs + tail_mass = 1 - sum(p_topk)** only at
  assistant positions (shifted -1). T=2 is applied later in the trainer.
- Heretic: 100% abliterated student; safety lives in `configs/scope.yaml`
  (default-deny) and harness checks, not in the weights. A second conservative
  abliteration pass only if the 10.6 refusal probe fails; never abliterate the
  final LoRA adapter.

## Run configuration (Phase 10.5)
- LoRA: r=32, alpha=64, dropout 0.05, targets q/k/v/o + gate/up/down,
  lr 2e-4, cosine, warmup 0.03, seq 2048, batch 1 × grad_accum 16, seed 7,
  AdamW, checkpointing on. Trainable params: 34,865,152.
- Both runs: `max_steps=1000`, eval every 100, checkpoint every 50.
  1322 records × 16 samples/step ⇒ ~12 fixed-order epochs by step 1000.
- Run A = text_sft (teacher text only). Run B = logit_kd
  (alpha 0.5, T=2, forward KL, top-32).
- Dtype: trainer loads bf16 (no GradScaler). DECISIONS item 3 expects
  fp16+GradScaler on T4. `_device_banner` now prints GPU + native bf16.
  Pending: verify speed/stability on the actual Colab GPU; implement scaler
  only after confirmation.
- Metrics: `metrics.jsonl` per session (step/loss/ce/kd/grad_norm/lr/eval_*);
  `summary.json` only at max_steps. Never merge session files.
- Checkpoints: local `runs/<run>/checkpoints/checkpoint-N` + outbox mirror.
  New saves are atomic (`.tmp-checkpoint-N` → rename + `COMPLETE` marker).
  Never delete `phase10_text_sft/checkpoints/checkpoint-1000` (it is A@800).

## Current state (edit me)
- **2026-10-21 external review (verified)**: logit_kd was inconsistent at T>1
  (teacher tail frozen at T=1, student tail from the full vocab at T=2 ⇒
  over-sharpening). Fix in `losses/logit_kd.py`: T=1 keeps the cached tail
  (exact); T>1 renormalizes BOTH sides over the top-32 support and drops the
  tail (exact over that support). Identity teacher==student ⇒ KL≈0 tested for
  T∈{1,2}. Config stays T=2.0 (top-32 semantics documented in the yaml).
  Real-cache tail stats: >1e-4 in 7.2% of 80,224 positions, mean 2.63e-4,
  max 0.3027 — the reviewer's "3.5 nats" was a synthetic worst case.
- **No-op guards added** (would have caught the A incident immediately):
  `load_training_state` rejects checkpoints whose dir step != optimizer
  internal step AND whose scheduler `last_epoch` != step; both loops log
  `grad_norm` and abort on non-finite; after the first `opt.step()` of a
  session a LoRA weight checksum must change. On resume the LR is recomputed
  as `base_lr * lr_lambda(step)` (protects hand-edited reconstructions whose
  saved lr is 0).
- **Atomic checkpoints**: local + outbox save via `.tmp-checkpoint-N` then
  rename + `COMPLETE` marker. Notebook resume cell validates dir step ==
  training_state step == optimizer step and quarantines invalid ones
  (`.invalid-checkpoint-N`, ignored by auto-resume). Legacy Drive checkpoints
  without marker are still accepted after validation.
- **Run A**: trained genuinely 0→800; the Drive file named
  `phase10_text_sft/checkpoints/checkpoint-1000` is the genuine 800-weight
  state (opt.step=800, eval loss 0.2657). Do NOT reconstruct it. Use it
  directly as **A@800**. If ever resumed, the fake label must be fixed
  (step/scheduler) or the new guards refuse; lr is recomputed anyway.
- **Run B**: stopped at **850** real steps (old inconsistent T=2 loss;
  checkpoint-800 and 850 saved/mirrored). B eval: ppl 5.2037 at 300,
  5.3257 at 700 (+2.3%, possible early overfit or noise). Plan: evaluate
  **B@800** as `B-legacy` against A@800 first (cheap, no training). If it wins
  clearly, use it; if it loses/ties, restart B from 0 with the fixed loss to
  800 (same cosine-to-1000 schedule) before concluding KD does not help.
  Restart cost: ~4 sessions; finishing old-B to 800 would be ~1.5, so ~2.5
  extra net. Before any restart: move Drive `runs/phase10_logit_kd` aside or
  the resume cell restores the old 800/850.
- **A vs B confound**: 281/1322 records differ (A uses `student_messages`;
  B uses the full teacher context because the cache indexed it). All 281 are
  the records without a system prompt. Eval both with and without the system
  prompt (notebook cells 12/13).
- **10.6-dev tooling**: `scripts/eval_dev.py` evaluates a merged model on the
  DEV split ONLY (sealed test stays untouched), reproducing BASELINE0_v2
  exactly on student_base: PPL 3.7026/40.553, identity 2/4/2, toolcall 5/10.
  It imports `suite_toolcall` from the harness. Harness now passes `--model`
  to the persona subprocess (it silently evaluated `student_base` before) and
  supports `--device/--dtype/--system-prompt`. Notebook cells 12/13: merge
  A800/B800, dev eval (A/B), then system-prompt variant; results → Drive.
  Do NOT use the full harness sets for the A/B decision: they include sealed
  test IDs. Sealed (93 hashed IDs) is evaluated once for the final winner.
- **Baseline/gates (BASELINE0_v2 dev)**: persona-dev 16/18/14 (n=48,
  0.4706); obedience 9/2 (n=11, 0.8182); identity 2/4/2 (n=8, 0.3333);
  injection 7/33 (n=40, 0.1750); devops 11/4/5 (n=20, 0.7333); security
  7/2/11 (n=20, 0.7778); sandbox 40/13 (n=53, 0.7547); toolcall 5/10;
  PPL-big mean 3.7026 / PPL 40.553 (60 lines, 48-token windows).
  Gates: persona auto ≥0.47 (Wilson lo ≥0.31); obedience/identity no new
  fails; injection ≥0.50; devops/security no significant drop; sandbox no
  new fails; toolcall ≥5/10 then non-decreasing; PPL-big within [3.51, 3.90];
  PPL-38 within +10% of 18.016; lm-eval n≥200 drop ≤0.03 (deferred to GPU).
  Only injection (n=40) has useful power; treat small suite deltas as noise.
- **Contamination**: 8-gram max overlap vs sealed test measured **0.0** on
  train_v1+dev_v1; the 0.5 threshold never bound (still lax in general).
- **If A beats B**: decide by gates first, then weight security/devops/
  tool-calls; use Wilson/bootstrap; if within noise prefer simpler A or
  replicate. No adapter merging/Frankenstein. An A win likely means B
  overfit → tune alpha/T or early-stop, not "KD useless".
- **Remaining 10.6 blockers**: refusal probe (~30–50 ES/EN bad prompts,
  gate "no more refusals than base") before merge/export; capability lm-eval
  on the winner; sealed test once; merge LoRA + Q4 GGUF export.
- **Admin pending**: push this commit (Ling); GPU hours in
  `docs/GPU_LEDGER.md`; external backup of the repo (~87GB); human eye-review
  of ~20 eval pairs; Drive rotation (`runs/checkpoints/` soup deletion after
  confirming); dtype decision above.
- Suite: **85 green** (`PYTHONPATH=src .venv/bin/python -m pytest tests -q`).
  Never run bare `pytest` at the repo root: it collects vendored llama.cpp
  tests and errors. README/HANDOFF_CLAUDE are not the source of truth.
- Next: pull this commit → notebook cell 12 (B-legacy@800 vs A@800) → if B
  loses/ties, restart B fixed to 800 and rerun → decide → merge/export Q4.

## Runtime / product decisions
- Serving: **llama.cpp/Ollama** for the local GGUF (single user, CPU-first).
  vLLM only for future serving on a 3090 (27B Turbo/Strata are hardware-
  blocked today).
- NVIDIA acquired Hugging Face (announced 2026-09-03, ~US$12.93B) but
  llama.cpp is independent (ggml-org, MIT). NVIDIA contributes upstream
  (MTP/speculative decoding, ~1.9× on RTX 5090); official RTX Spark recipes
  use llama.cpp + Qwen3.8-27B Q4_K_M + DFlash draft. No reason to migrate.
- DeepSeek's DeepInfer is an inference framework (xgrammar), their V4.1
  Flash/Engram is a 552B frontier model; neither replaces our dataset/persona/
  pipeline/harness. Reuse only ideas/code (e.g. xgrammar at serving).
- Tool-calls: do NOT use constrained decoding to fix the eval schema score
  (it would measure the decoder). The runtime may use trigger-gated grammar
  (KMP; only inside a tool-call block), never global JSON grammar.
- Qwen3-Coder-30B-A3B-Instruct is tokenizer-compatible but ~18GB NF4 does not
  fit a T4 16GB, and it is code-biased: not a teacher for this dataset.
  If 8B is insufficient, Qwen3-14B/32B/30B-A3B (general) are the candidates,
  with a fresh cache extraction.

## Hard rules
- Local open teachers only (Apache-2.0/MIT, license + fingerprint recorded).
- Defensive + authorized scope only (`configs/scope.yaml`, default-deny).
- 100% heretic: no refusal training; limits live in code, not weights.
- Rejected teachers (ToS): Grok, Claude — reference only.
- Push needs Ling (SSH passphrase); Colab clones fresh after history rewrites.
- Do not commit secrets, tokens, `models/`, `student_base/`, `runs/`, or
  Drive paths outside this file.
