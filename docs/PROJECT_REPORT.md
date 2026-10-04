# PROJECT REPORT — ayanami-distill (all phases, consolidated)

Date: 2026-10-04. Student: `orlandorubino/Qwen3-1.7B-heretic` (Qwen3-1.7B
abliterated, BF16). Branch: `cpu_only` (HP EliteBook 845 G8, no usable GPU).
All claims below point to evidence files in this repo.

## Phase 0 — Preflight and workspace: DONE
- Machine inventoried from live output: Arch, Ryzen 5 PRO 5650U 6C/12T
  (AVX2, no AVX-512/AMX), 14.9 GiB RAM, 151 GB free disk, no GPU.
- Branch `cpu_only`; training deferred to `remote_gpu`. Repo initialized.
- Evidence: `docs/ENV_INVENTORY.md`, `docs/DECISIONS.md`, `.gitignore`.

## Phase 1 — Acquire and audit: DONE
- `student_base/` = verified read-only copy (SHA256 match, `chmod -w`,
  upstream rev `433869992d…`, file set == upstream sibling list).
- 1,720,574,976 params BF16, 310 tensors, 0 NaN/Inf; 28 layers, 2048 hidden,
  16/8 heads, vocab 151936, tied; tokenizer fingerprint `563a701b…`;
  ChatML+thinking+Hermes template renders; 20-prompt baseline (3.78 tok/s).
- VERIFY: layers/hidden/heads/vocab/tied/long-context/transformers-5.x/
  thinking/tool-calls CONFIRMED; lineage/refusals/KL/languages = card claims.
- Evidence: `docs/STUDENT_AUDIT.md`, `configs/student.yaml`,
  `scripts/audit_student.py`, `scripts/smoke_baseline.py`.

## Phase 2 — Python environment: DONE
- `.venv` Python 3.12.15 (uv standalone; system 3.14 untouched); 99-line
  `requirements.lock` (torch 2.14.1+cpu, transformers 5.18.0, peft, trl,
  datasets, lm-eval 0.4.13, optuna; no bitsandbytes); `check_env.py` PASSES
  (fwd loss 8.6044, bwd grads on 310/310 tensors; full backward 129.8 s —
  empirical support for `cpu_only`); `setup_remote.sh` for GPU machines;
  system llama.cpp 0.5.0-dev b11146 confirmed.
- Evidence: `requirements.lock`, `scripts/{check_env.py,setup_remote.sh}`.

## Phase 3 — Persona contract: DONE
- `identity_facts.yaml` (Ayanami, creator Ling, AI nature, cybersec+DevOps,
  en+es), `system_prompt.md` (Appendix A + 1 typo fix), 72-case eval set
  (EN37/ES35) + scorer; untouched baseline 25 pass / 24 fail / 23 review.
- Evidence: `persona/`, `src/ayanami_distill/eval/persona_eval.py`.

## Phase 4 — Teacher layer: DONE (17 tests green)
- Adapter interface (4 backends), Appendix-B record schema + validation,
  mode router with reasons, identity scrubber (tested), top-k cache with
  disk estimator, license/ToS gate, fake teacher + fixture.
- Evidence: `src/ayanami_distill/{teachers,data/records}.py`,
  `tests/test_phase4.py`, `configs/teachers/fake.yaml`.

## Phase 5 — Training scaffold: DONE (14 tests green)
- 4 loss modules (text_sft masks verified on real tokenizer; logit_kd
  α=0.5/T=2/fwd-rev-JSD exact at T=1; own ULD cross_tokenizer — trl 1.14.1
  VERIFIED to lack GKD/GOLD; hidden projector); LoRA targets VERIFIED in
  checkpoint; trainer CLI (seeds, ckpt/resume, drift PPL, run_meta);
  on-policy stub.
- Smoke: 50-ex run (loss −72%, adapter roundtrip) + focus run (6 facts,
  loss −99.96%, 3/3 outputs changed, 2/3 exact hits "Ling is my creator."
  / "done."; miss "Ling is my name." = documented tiny-data conflation).
- Evidence: `src/ayanami_distill/{losses,train,data/mix}.py`,
  `configs/{lora,distill/*}.yaml`, `scripts/smoke_train.py`,
  `tests/test_losses.py`, `tests/test_train_scaffold.py`.

## Phase 6 — Tool-use readiness: DONE (suite 39/39 green)
- 13-tool registry (12 read-only + scope-gated port_scan), Hermes validator,
  default-deny `scope.yaml` (localhost only), trajectory→record converter,
  10 hand-made trajectories (all validate; tool outputs provably untrained).
- Evidence: `src/ayanami_distill/tools/`, `configs/scope.yaml`,
  `tests/test_phase6.py`.

## Phase 7 — Eval harness + baseline-0: DONE
- One command, 7 suites: persona .51, obedience .75, devops .77,
  security .57, injection 2/8/2, toolcall parse 1.0/schema 0/correct 0,
  PPL 18.016, arc_easy 0.633 / boolq 0.867 (0-shot, n=30). 7 thresholds
  proposed (Ling approves). 2 harness bugs found+fixed during the run.
- Evidence: `src/ayanami_distill/eval/harness.py`, `docs/BASELINE0.md`,
  `runs/eval0/eval.json` (gitignored).

## Phase 8 — Export loop: DONE
- Streaming merge (56 tensors patched, diff 0.000774); vendored converter
  at pinned commit; F16/Q4_K_M/Q8_0; template byte-identical;
  BF16 2/3 @2.39 tok/s vs Q4 2/3 @17.8 tok/s vs Q8 2/3 @11.8 tok/s —
  identity+math survive quantization exactly. Q4_K_M = deployment format.
  llama-cli of this vintage needs `-st` (interactive REPL default; cost us
  a 3 h hang, fully diagnosed). Thinking-off default recorded.
- Evidence: `scripts/{merge_lora,export_smoke}.py`, `scripts/vendor/`,
  `exports/*/`, `docs/RUNBOOK.md` (export chapter).

## Phase 9 — Handoff: DONE
- `README.md` (purpose, status, map, setup, continue-list, safety),
  finished `docs/RUNBOOK.md`, teacher rulings logged, 39/39 green.
- Evidence: this file + `README.md` + `docs/RUNBOOK.md`.

## Teacher rulings (verified, not from memory)
- **Pingu Unchained: REJECTED** — commercial API-only (waitlist/ID/pay),
  ToS unverified→gate refuses, offensive/poisoned mission vs defensive scope,
  no logit compatibility, high scrub burden.
- **Qwen/Qwen3-8B (Instruct): PROPOSED, pending Ling** — Apache 2.0;
  tokenizer fingerprint byte-identical to student (computed, `563a701b…`);
  fits free T4 in 4-bit; Instruct follows/tools/reasons; Qwen identity
  already covered by tested scrubber. Gate before caching: beat baseline-0
  on 80 domain cases + toolcall.

## Known risks
- Zero real distillation has run: all "trained" artifacts are smoke toys.
- CPU step cost (~8 min) makes local training infeasible; `remote_gpu` access
  is the critical path. 3 commits (9d5c64b, a2999b7, 5a945bd) pending push
  (ssh-agent empty after reboot).
- 23+ manual review cases, 80 domain references, 7 thresholds, scope assets
  await Ling.

## Open questions for Ling
1. Approve Qwen3-8B-Instruct as first teacher? 2. GPU access, when?
3. Approve thresholds? 4. Scope assets + domain reference review?
5. Re-add ssh key so the 3 commits can be pushed?
