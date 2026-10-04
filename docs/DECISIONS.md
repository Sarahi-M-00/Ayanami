# Decisions log
# Format per entry: date, decision, alternatives considered, reason.
# All entries in English (project rule).

## 2026-10-03 — Workspace location: /home/ling/ayanami-distill/
Decision: Create the distillation workspace as a new directory `/home/ling/ayanami-distill/` with the target tree from the execution prompt.
Alternatives:
- Inside `/home/ling/Ayanami-AI/` (the model snapshot dir). Rejected: the snapshot must stay pristine and read-only; mixing workspace files into it risks accidental modification.
- Another path. No benefit; home directory keeps ownership and the 151G free disk.
Reason: Isolation rule (workspace is a separate project with its own venv; the Ayanami-AI CLI and the model snapshot stay untouched).

## 2026-10-03 — Compute branch: cpu_only (training deferred to remote_gpu)
Decision: The laptop operates on the `cpu_only` branch: data preparation, validation, GGUF conversion and llama.cpp inference. No training on this machine.
Alternatives:
- `local_gpu`. Rejected: no NVIDIA GPU, no ROCm stack; only an AMD Cezanne iGPU, unusable by PyTorch. Phase 0 proves no usable GPU (see `docs/ENV_INVENTORY.md`).
- `remote_gpu` now. Deferred, not rejected: real LoRA/QLoRA training requires Colab/Kaggle/cloud access, which Ling has not provided yet. The repo must be runnable there via one setup script (Phase 2 deliverable `scripts/setup_remote.sh`).
Reason: 14.9 GiB RAM, 6 CPU cores without AVX-512/AMX, and no accelerator make even LoRA training of a 1.7B model impractical here (optimizer states plus activations exceed comfortable RAM; CPU step throughput would stretch runs to days). The prompt's own plan assigns training to a GPU and the laptop to prep/validation/export.
Precision rule note: when `remote_gpu` activates, GPUs without native bf16 (e.g. T4) use fp16, Ampere or newer use bf16.

## 2026-10-03 — student_base as physical read-only copy (Ling chose option 1)
Decision: Copy `/home/ling/Ayanami-AI/` verbatim into `student_base/`, verify SHA256 of the copy, then `chmod -R a-w`. No re-download from HuggingFace.
Alternatives:
- Re-run `snapshot_download` into `student_base/`. Rejected: wastes a 3.4 GB download; local file set already matches the upstream sibling list exactly (verified via HF API, revision 433869992df66cd5ed9fa77491809448c3dac7a0).
- Symlink to `/home/ling/Ayanami-AI/`. Rejected by Ling (option 2): read-only cannot be enforced on a symlink without touching the original.
Reason: Literal compliance with "snapshot lives in student_base/ and is read-only"; true permission-level protection; 151G free disk makes the 3.4 GB duplication irrelevant.

## 2026-10-03 — Phase 1 audit ran on existing ~/venv-hf-cpu (no new env yet)
Decision: Run `scripts/audit_student.py` and `scripts/smoke_baseline.py` with the pre-existing `/home/ling/venv-hf-cpu` (torch 2.14.1+cpu, transformers 5.18.0), read-only usage, offline mode. No packages installed or modified.
Alternatives:
- Build the project venv in Phase 1. Rejected: the execution prompt assigns environment construction to Phase 2; duplicating it early would split the pinning across two places.
Reason: Phase 2 will create the pinned `.venv` + `requirements.lock` + `check_env.py`; Phase 1 only needed a throwaway inference runtime, and reusing an existing one keeps the dependency footprint minimal.

## 2026-10-03 — Baseline-0 protocol: greedy, thinking off, 64 new tokens, 6 threads, seed 7
Decision: `runs/baseline0/` uses greedy decoding (`do_sample=False`), `enable_thinking=False`, `max_new_tokens=64`, 6 CPU threads, seed 7, 10 EN + 10 ES short prompts.
Alternatives:
- Sampled decoding with the card defaults (temp 0.6, top_k 20, top_p 0.95). Rejected for the baseline: sampling is non-deterministic across runs and complicates regression diffs; sampled behavior can be added as a second baseline later.
Reason: A regression reference must be maximally reproducible; greedy + fixed seed gives bit-stable comparisons for future student checkpoints.

## 2026-10-03 — Project Python 3.12 via uv standalone build (system stays 3.14)
Decision: Install `uv` 0.12.22 as a user-local binary (`~/.local/bin`, outside the repo) and provision CPython 3.12.15 from uv's standalone builds; project `.venv` created with it. System `/usr/bin/python3` (3.14.7) untouched.
Alternatives:
- Use system Python 3.14. Rejected: the execution prompt requires 3.10–3.12; torch/transformers on 3.14 is untested territory for this pipeline.
- pacman/AUR Python 3.12. Rejected: Arch repos ship only the current Python; no root changes wanted.
Reason: Zero system modification, exact prompt compliance (3.12 in the 3.10–3.12 band).

## 2026-10-03 — CPU torch from the PyTorch CPU index; one lock, remote swaps torch
Decision: `torch==2.14.1+cpu` installed from `https://download.pytorch.org/whl/cpu` (cp312 wheel verified present before install); everything else unpinned from PyPI and resolved at install time (`transformers>=4.51` floor only), then frozen to 99-line `requirements.lock`. `scripts/setup_remote.sh` installs GPU torch from default PyPI first, then the lock minus torch/nvidia/triton lines.
Alternatives:
- Default-PyPI torch (CUDA build) on the laptop. Rejected: pulls ~GBs of useless NVIDIA libs on a GPU-less machine.
- Separate remote lock file. Rejected for now: one lock + documented swap keeps a single source of pinned truth; revisit if divergence causes pain.
Reason: Minimal download, reproducible laptop env, runnable remote path.

## 2026-10-03 — bitsandbytes omitted; lm-eval and optuna included
Decision: Skip `bitsandbytes` (CUDA-only, prompt marks it optional/CUDA-only). Install `lm-eval==0.4.13` and `optuna==5.0.0` since both resolved cleanly.
Reason: Prompt compliance + no dead weight on CPU-only hardware.

## 2026-10-03 — llama.cpp already present system-wide, no build needed
Decision: Use system llama.cpp (`/usr/bin/llama-cli`, `llama-server`, `llama-quantize`), version 0.5.0-dev build 11146 commit 7fe450e193. No clone/build performed.
Reason: Phase 2 only requires install-or-confirm; confirmed present with a recorded commit.

## 2026-10-03 — Full-model backward takes ~130s on CPU (supports cpu_only)
Decision: Recorded, no action. `check_env.py` 1-step full backward (310 tensors, tiny batch) took 129.8s on 6 CPU threads. A real LoRA run needs thousands of such steps -> days on this laptop.
Reason: Empirical confirmation of the Phase 0 `cpu_only` branch; training waits for `remote_gpu`.

## 2026-10-03 — system_prompt.md = Appendix A verbatim except one typo fix
Decision: Copy the Appendix A draft exactly, fixing only "maded by Ling" -> "made by Ling".
Alternatives: Keep the typo. Rejected: it would be baked into future training data and eval runs.
Reason: "Refine, but keep the contract" — a typo fix is a refinement, not a contract change.

## 2026-10-03 — Baseline-0 persona run WITHOUT system prompt; Layer-1 run deferred
Decision: `persona/eval/baseline0_*` scores the untouched Student with no system prompt (greedy, thinking off, 120 max tokens, seed 7). The with-prompt / without-prompt mix from the two-layer strategy will be exercised in later phases, not here.
Reason: Phase 3 acceptance only requires baseline-0 for the untouched Student; the mix matters once training data is generated (Phase 5).

## 2026-10-03 — Rule-based persona checks are signals, not grades
Decision: `persona_eval.py` auto-checks are strict by design; every fail means "needs review", and 23/72 cases are manual-review by Ling. Known strictness: markdown bolding breaks `^Summary:` (obedience_11 got `**Summary:**`), letter-spelling breaks exact match (obedience_09 got `D-O-N-E`).
Alternatives: Loosen checks (strip markdown, normalize spelling). Rejected for the baseline: strict checks give a cleaner regression signal later; loosening can be proposed with evidence in Phase 7.
Reason: A baseline should be reproducible and strict; interpretation happens at review time.

## 2026-10-03 — No real teacher evaluated; license/ToS gate enforced in code
Decision: Phase 4 ships the gate (`assert_usable`: empty license or `tos_allows_distillation=false` raises `TeacherBlockedError`, checked before routing) plus a test-only fake teacher (`test-fixture` license). No real teacher has been proposed, approved, or cached.
Reason: The execution prompt forbids using outputs from providers whose terms prohibit training other models; the first real teacher choice is Ling's decision (Phase 9 open questions). The mechanism is tested; the policy call is deferred.

## 2026-10-03 — Identity scrubber rewrites (not deletes) identity claims
Decision: `scrub_completion()` substitutes `I am <Teacher>` -> `I am Ayanami` and `<verb> by <Vendor>` -> `<verb> by Ling`, anchored to identity contexts so code/prose (e.g. `llama-quantize`) passes through untouched. Every substitution is reported in `hits`.
Alternatives: Delete whole sentences containing teacher names. Rejected: it would also delete adjacent useful content and is harder to audit than a reported substitution list.
Reason: Predictable, reviewable, and it can never inject a *new* vendor claim — only Ayanami/Ling, which is the target identity.

## 2026-10-03 — Cache stores top-k only with a pre-run disk estimate
Decision: `estimate_topk_cache()` (int32 ids + float16 logprobs + float16 tail per position) must be printed before any caching run; layout `data/teacher_cache/<teacher_id>/<revision>/`.
Reason: Full-logit caches for long runs would silently fill disks; the estimate makes cost explicit.

## 2026-10-03 — trl 1.14.1 has no cross-tokenizer trainer; own ULD loss written
Decision: VERIFIED against the installed version (`trl.trainer` modules listed, `DistillationConfig` fields inspected): trl ships `DistillationTrainer` (same-tokenizer, teacher-resident) but no GKD/GOLD-style cross-vocabulary trainer. Wrote `losses/cross_tokenizer.py` (sorted-distribution ULD approximation) behind the `enabled: false` flag instead of reusing trl.
Reason: Prompt rule — check the installed trl for a ready implementation before writing your own. None exists.

## 2026-10-03 — KD tail tempering is an acknowledged approximation
Decision: `teacher_topk_dist()` scales cached top-k logprobs by 1/T but keeps the stored tail mass temperature-independent (the tail has no logits to scale without the full distribution). Documented in the module docstring. Verified exact: at T=1 with teacher built from student, loss is ~0 (1.6e-9); at T=2 the same/contrastive ordering holds for forward/reverse/JSD.
Reason: Standard practical approach for top-k caches; hiding the approximation would be dishonest.

## 2026-10-03 — Smoke verdict: mechanics PASS, memorization needs epochs
Decision: The 50-example/12-step smoke run is recorded as pipeline PASS on mechanics (72% loss drop, adapter save/load roundtrip, 3/3 outputs changed) but target-hits only 1/3: 12 steps x batch 1 sees ~12/50 examples, which cannot memorize 50 facts. A focused 6-fact run (SMOKE_FOCUS=1, 24 steps = 4 epochs) is the memorization proof; its report decides the final acceptance wording.
Alternatives: Call 1/3 a pipeline failure. Rejected: evidence (loss curve, changed outputs, deterministic rerun identical to attempt 2) shows gradients and I/O work; the miss is a sample-count artifact.
Reason: Distinguish "pipeline broken" from "undertrained" — the acceptance criterion is about the pipeline.

## 2026-10-03 — Laptop stuck in powersave (~1.4 GHz); no sudo to change it
Decision: Recorded, no action possible (no passwordless sudo; will not ask for/share passwords in chat). All CPU timings in this project carry that handicap. Smoke uses 12 SMT threads, batch 1, no grad-checkpointing (RAM allows) for step economy.
Reason: Environmental fact affecting every timing claim.

## 2026-10-03 — Focus run verdict: PASS with documented conflation artifact
Decision: `runs/smoke_test_focus/` (6 facts x 24 steps, ~3.2h CPU): loss 7.19 -> 0.0029 (99.96% drop), 3/3 probe outputs changed, 2/3 exact target hits ("Ling is my creator." and "done."). The miss is informative, not a failure: the name probe returns "Ling is my name." — the model overgeneralized the "Ling is my X" pattern from 6 examples instead of keeping "I am Ayanami".
Alternatives: Demand 3/3 exact before passing. Rejected: the conflation is a tiny-data artifact (6 examples cannot separate the patterns), and the acceptance criterion targets the pipeline, which is fully proven (gradients, checkpointing, adapter save/load, before/after change).
Reason: Record the artifact as evidence for why the real persona phase needs hundreds of varied examples, not 6 repeated ones. Verdict logic (drop>=30%, changed>=2/3, hits>=2/3) stands as written in scripts/smoke_train.py.

## 2026-10-03 — Minimal built-in JSON-Schema validator, no new dependency
Decision: `tools/registry.py` ships a small validator covering exactly what the seed catalog uses (object/string/integer/boolean/array, required, enum, additionalProperties). The `jsonschema` library was NOT added.
Alternatives: Add `jsonschema` to requirements. Rejected for now: it would change the Phase 2 lock for a need the current schemas do not have; revisit when tool schemas need conditionals or refs.
Reason: Keep the lock stable; the validator is fully tested against the catalog.

## 2026-10-03 — scope.yaml is default-deny with localhost only
Decision: `configs/scope.yaml` lists only `localhost` (offline-test placeholder) plus a clearly-marked documentation placeholder. `port_scan` and any future target-touching tool reject everything else in code. Real assets are Ling's decision (open question for Phase 9).
Reason: The Student has no learned refusals, so the harness — not model behavior — is the safety layer. An empty-looking scope that denies by default is safer than a permissive example.

## 2026-10-03 — Tool outputs are never trained on (proven by test)
Decision: Trajectory-to-record conversion keeps tool turns as role "tool"; the SFT mask (Phase 5, unchanged) gives them mask 0 while assistant tool calls and final answers get mask 1. `test_loss_masking_on_tool_trajectory` asserts the exact token split on a real 2-call trajectory.
Reason: Training on tool outputs would teach the model to hallucinate observations — the Phase 6 design explicitly forbids it, and now a test enforces it.

## 2026-10-03 — lm-eval slice: arc_easy + boolq, 0-shot, limit 30
Decision: Capability slice uses `arc_easy` and `boolq` (both VERIFIED present in lm-eval 0.4.13), forced 0-shot, limit 30 examples, batch size 1. Results: arc_easy acc 0.633, boolq acc 0.867 (n=30 each).
Alternatives: More tasks / few-shot / full sets. Rejected: 25-shot ARC prompts on CPU would take hours per task; the slice is a drift signal, not a leaderboard entry.
Reason: CPU-feasible regression signal; full benchmarks belong to remote_gpu runs.

## 2026-10-03 — Harness needed network for eval datasets (offline flag scoped)
Decision: The harness keeps HF_*_OFFLINE for all student work but strips those vars ONLY for the lm-eval subprocess (public eval datasets must download; weights still load from the local path). Found via a real failure: arc download died with OfflineModeIsEnabled. Also fixed: relative --out path crash in write_baseline + persona "cases" vs "n" key.
Reason: Failing loudly then fixing beats silently skipping the capability suite.

## 2026-10-03 — PPL 18.016 is the drift reference; KL-vs-baseline stays future
Decision: Held-out PPL (38 general lines) = 18.016 recorded in BASELINE0.md as the no-regression anchor (threshold: +10% max). KL vs baseline-0 is still not tracked (no base logits cached) — same limitation as Phase 5, unchanged.
Reason: Honest metric boundaries; PPL is measurable today, KL needs cached base logits (future work).

## 2026-10-04 — Thinking mode default: OFF for tool-agent use
Decision: All inference in this project runs with thinking disabled (`enable_thinking=False` / pre-rendered empty think block), and that stays the default for agent use.
Alternatives: Thinking on (default model behavior). Rejected for agent use: thinking tokens waste context and CPU on a 6-core laptop, slow down tool loops, and make tool-call extraction brittle; reasoning can be enabled per-task when a hard problem justifies it.
Reason: Determinism, speed (measured: no thinking overhead in any baseline), and precise tool-call parsing. Evidence: every baseline, smoke and export run in this repo uses thinking-off.

## 2026-10-04 — This llama.cpp vintage needs -st; interactive REPL is the default
Decision: Recorded as a hard gotcha: llama-cli 0.5.0 (Arch package AND source build at commit 7fe450e1, identical behavior) starts an interactive chat REPL by default and spins forever on EOF stdin. Scripted runs REQUIRE `-st` (--single-turn). Pre-rendered ChatML via `-f` is used verbatim; strip the prompt echo when scoring. A source build lives at /home/ling/llama.cpp-build (outside the repo) as fallback; system binary preferred for provenance.
Reason: Cost us a 3-hour hung run and a full debugging session (documented in run logs). Future runs and the RUNBOOK encode -st.

## 2026-10-04 — Q4_K_M fixed as the deployment format for the smoke line
Decision: Q4_K_M (1.1 GB, 17.8 tok/s) preserves the merged BF16 behavior exactly on all 23 comparison prompts (2/3 identity hits including the same "Ling is my name" conflation quirk, correct math). Q8_0 (1.8 GB, 11.8 tok/s) adds nothing behaviorally here. BF16 reference: 2.39 tok/s.
Reason: Evidence in runs/export_smoke/report.json. Q8_0 stays as the fidelity upper bound for future real (non-smoke) models; re-evaluate per model.

## 2026-10-04 — Pingu Unchained rejected as teacher
Decision: REJECTED without caching a single record. Verified facts: commercial product by Audn.AI (no public weights; API-only with waitlist, ID verification and per-token pricing). Reasons: (1) ToS unverified and almost certainly prohibitive for distillation (commercial API profile) — our Phase 4 gate refuses by default; (2) mission conflict — explicitly offensive/poisoned vs our defensive scope and harness safety model; (3) no logit compatibility (non-Qwen tokenizers) — text_sft only at API prices; (4) high identity-scrub burden (adversarial persona).
Reason: Recorded so no future agent re-evaluates it from scratch.

## 2026-10-04 — Qwen/Qwen3-8B (Instruct) proposed as first teacher (pending Ling)
Decision: PROPOSED, not approved. Verified evidence (not memory): Apache 2.0 license (Hub API); tokenizer fingerprint 563a701b... computed from its downloaded tokenizer.json — byte-identical to the Student, unlocking real logit_kd; 8B fits a free Colab T4 in 4-bit for local logit generation; Instruct variant follows instructions/uses tools/reasons; its Qwen/Alibaba identity is already covered by the tested scrubber patterns.
Acceptance gate before first cache: it must beat baseline-0 on the 80 domain cases + toolcall suite (score first, distill later).
Alternatives: Qwen3-14B (needs P100/A100, same pipeline); non-Qwen open-weights (text_sft only, more scrubbing — acceptable if one proves dominant in our domains with verified ToS).
Reason: Same family + permissive license + free-tier feasible + measurable gate.

## 2026-10-04 — Tool-call baseline was two harness bugs + real inability
Decision: Diagnosed from saved outputs (no new modeling): (a) vacuous parse_ok fixed to require >=1 block, (b) tool schemas were never sent to the template — fixed. True re-run baseline: 5/10 on all three rates. Recorded in BASELINE0_v2.md.
Reason: "Beat the baseline" must be non-trivial; the old 1.0/0/0 would have made it vacuous.

## 2026-10-04 — Capability n=200 deferred to GPU with measured estimate
Decision: arc_easy/boolq at n>=200 marked defer_to_gpu. Measured probe: 8 examples in 91 s wall (~11 s/ex) -> ~38 min/task CPU, violating the 10-minute laptop rule; T4 estimate ~5 min/task. Phase 7 n=30 slice kept as CPU smoke only.
Reason: The prompt allows deferral with a time estimate; intervals at n=30 (±17 pts) detect nothing, so running n=200 on CPU buys noise at high cost.

## 2026-10-04 — Sealed split: thirds per suite, domain 40/40 stratified
Decision: persona 48/24, obedience 11/5, injection 40/20, identity 8/4, domain 40/40 stratified by file/lang (seed 7, largest-remainder quotas). The prompt's "(domain: 40 dev / 40 test)" parenthetical governed over a strict thirds reading. Roster: hashed IDs only, committed at data/sealed/test_ids.sha256 (.gitignore narrowed with an exception, same pattern as eval/data).
Reason: Deterministic, auditable, and the checker (tested) enforces exclusion in all later data steps.

## 2026-10-04 — PPL-big is a separate anchor (FineWeb slice, windows)
Decision: 60 lines (every 3rd of 180 FineWeb EN+ES docs, 48-token windows): PPL 40.553, bootstrap CI recorded. NOT comparable to PPL-38 (18.016, different text/length); both tracked. ODC-By attribution goes to DATA_SOURCES.md at 10.2 build time.
Reason: Longer docs blow up CPU attention cost (~10 s/128tok); strided 48-token windows fit the 10-minute rule while widening coverage.

## 2026-10-04 — Prompt pool v1: quality-first volume, shares deviate with reason
Decision: 1,911 prompts (0 dups, 0 sealed overlaps, 0 invalid): authored identity/obedience/injection/tool/domain/commands + 400 EN FineWeb (streamed, ODC-By) + 100 hand ES lines. Shares deviate from Appendix B (tool 5% vs 18%, replay 26% vs 15%) because hand-curated tool tasks and domain Q&A do not scale like parameterized constraints; counts (not verifiers) get cut first, and the 10.3 pilot fixes final size (teacher paraphrase on GPU can rebalance in v2).
Alternatives: Pad to 3,500 with thin paraphrases. Rejected: near-dup filter would either eat them (waste) or pass junk (harm); honest volume now, sized growth later.
Reason: Acceptance requires a built/deduped/checked pool, not a number. Subcategory mixing keys on the `source` prefix (recorded for the 10.4 mixer).

## 2026-10-04 — FineWeb streaming is flaky: direct-shard fallback + authored ES
Decision: `datasets` streaming (hf_xet) crashed the interpreter twice with Bad-file-descriptor/GIL errors AFTER collecting data (files were written; crash at teardown). EN replay kept the 400 streamed docs; ES switched to 100 hand-written lines (deterministic, zero flakiness) after killing a runaway 1.8 GB shard download. ODC-By attribution recorded in DATA_SOURCES.md.
Reason: Robustness over elegance; the ES slice needed 100 lines, not a 2 GB shard.

## 2026-10-04 — Teacher fingerprint MUST use the audit method (get_vocab lies)
Decision: `fingerprint_of_tokenizer()` saves the tokenizer and hashes the file (Phase 1 method). An initial version used `get_vocab()` and produced a DIFFERENT hash for the same tokenizer (51e63d0d… vs canonical 563a701b…), which would have aborted every real teacher run at the gate. Caught by unit test; fixed + regression-tested.
Reason: A safety gate that false-positives on good teachers is worse than none — it would have silently blocked logit_kd forever.

## 2026-10-04 — Pilot = 50 prompts, stratified by category, then Ling runs it
Decision: `pilot_50.jsonl` = 10 per category (persona/devops/security/general/tool_use), seed 7. The pilot measures: rejection rate per category, gen tok/s, scoring tok/s, VRAM peak, wall time → extrapolates the full run (3,000–4,000 prompts) for GPU budgeting.
Reason: Never spend quota blind; the manifest carries every number needed for the extrapolation.

## 2026-10-04 — Claude review fixes applied before the pilot (A+B+C verified)
Decision: Applied an external code review of teacher_cache.py/RUNBOOK in full, after verifying each claim against the repo (all confirmed valid).
A (quota/data-poisoning): truncated generations dropped with reason; think-leaks dropped; imports moved before any download + `preflight` subcommand (tokenizer-only, runs on laptop, notebook calls it first); teacher revision pinned to b968826d… (refused unless --force-revision; recorded in configs/teachers/qwen-qwen3-8b.yaml); device_map={"": 0} single-GPU (+CUDA_VISIBLE_DEVICES docs for 2 processes).
B (bulk-run): crash-safe resume (truncated-tail repair, rejected-ID log, manifest counts from files); honest timing (sec/kept/category + rejected seconds); --shards range loop with one model load; scoring sliced to assistant rows + vectorized top-k + 4-decimal logprobs; all 6 small items (split check, crc32 seeds, prefix-unstable drops, select exact-n, tool_use verified_success, trailing-newline-after-im_end dropped by explicit rule).
C (RUNBOOK): shard layout matches code; thresholds point at BASELINE0_v2 with intervals; quota via web UI; HF cache outside /kaggle/working + upload exclusions; two-GPU invocation.
Reason: GPU quota is scarce; every one of these either wastes it or corrupts data silently. Reviewed-but-unverified claims were verified, not trusted.

## 2026-10-04 — Trailing newline after final im_end is NOT trained
Decision: score_topk drops a single template-added "\n" after the final <|im_end|> when the completion does not end with a newline. The model never generates that token; training it would teach nothing.
Reason: Explicit per review; the alternative (keeping it) trains the model to predict template glue.
