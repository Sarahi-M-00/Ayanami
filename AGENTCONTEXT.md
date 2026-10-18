# AGENTCONTEXT — ayanami-distill (living file, keep current)

> Technical continuity only. Personal context lives OFF-repo (Ling's OneDrive).
> Update this file on: phase close, new decision, new tool, new teacher,
> pipeline change, or anything a fresh agent needs on day one.

## Mission
Build **Ayanami**: local 1.7B assistant (Rei Ayanami persona, loyal to Ling),
defensive cybersecurity + DevOps, via distillation. Student base:
`orlandorubino/Qwen3-1.7B-heretic` (abliterated, Apache-2.0).

## Where things are
- Code: `src/`, pipeline scripts: `scripts/`, configs: `configs/`.
- Dataset v1: `data/processed/train_v1.jsonl` (1322) + `dev_v1.jsonl` (67).
- Teacher cache: `data/teacher_cache/qwen-qwen3-8b/<rev>/` (20 shards).
- Training: `configs/distill/phase10_*.yaml` → `runs/`; Colab: `notebooks/colab_train.ipynb`.
- Docs: `docs/DECISIONS.md`, `docs/PHASE10_DECISIONS.md`, `docs/RESEARCH_IDEAS.md` (R-01..R-03).
- Hoard (gitignored): `models/`, `reserva-rea/` (REA/Longinus mirror + install.sh).
- Local runner: `llamita.cpp/` snapshot + `~/bin/llm` (Super+Shift+I).

## Current state (edit me)
- Phase: **10.5** — Run A (text_sft) COMPLETE: step 1000, eval 0.2657, ppl 1.30
  (checkpoint-1000 in Drive `runs/phase10_text_sft/checkpoints/`).
- Run B (logit_kd) incident (2026-10-17): every resumed session was a silent
  no-op (weights never changed; init eval stuck at exactly 1.6103244414282731).
  Two causes fixed in trainer.py: (1) `PeftModel.from_pretrained` defaults to
  `is_trainable=False`, which freezes the adapter on current PEFT (0.21.2) —
  now passes `is_trainable=True` + explicit no-trainable-params error; (2) the
  optimizer was built BEFORE the resume rebinding — now built after, with
  `_assert_optimizer_matches` guard (+2 regression tests). Drive 50-300 were
  deleted by rotation; 400/450 remain and are copies of the real step-150
  weights, so local `checkpoint-150` was reconstructed from 400 (step and
  scheduler reset to 150; optimizer state is the genuine 150 momentum).
  Audit pending: late Run A resume sessions may have no-op'd the same way —
  check `runs/phase10_text_sft/metrics.jsonl` on Drive (loss must decline
  *within* a session, after its init_eval row).
- Pending plan (agreed with Ling, do after Run B): automate the resume ritual —
  each Colab session starts empty, currently manual `cp` of latest checkpoint from
  Drive `runs/<run>/checkpoints/` to local `runs/<run>/checkpoints/` before launch.
  Options: notebook cell auto-pulls latest from Drive, or trainer reads outbox dir.
- Last verified: suite 79 green (`pytest tests/ -q`).
- Next: stop the current no-op Colab session → push fix → resume B from 150 →
  finish B → 10.6 compare + evals → merge/export Q4.

## Hard rules
- Local open teachers only (Apache-2.0/MIT, license + fingerprint recorded).
- Defensive + authorized scope only (`configs/scope.yaml`, default-deny).
- 100% heretic: no refusal training; limits live in code, not weights.
- Rejected teachers (ToS): Grok, Claude — reference only.
- Push needs Ling (SSH passphrase); Colab clones fresh after history rewrites.
