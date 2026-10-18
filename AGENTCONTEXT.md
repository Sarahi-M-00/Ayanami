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
- Run B (logit_kd) incident (2026-10-17): optimizer was built BEFORE the resume
  rebinding in trainer.py, so every resumed session was a silent no-op (weights
  never changed). Evidence: init eval 1.6103244414282731 identical when resuming
  from checkpoint-300 and checkpoint-450. Fixed: optimizer/scheduler now built
  after the rebind + `_assert_optimizer_matches` guard (+2 regression tests).
  **Resume Run B from checkpoint-150 (last real weights; 200..450 are duplicates
  of it — verify/delete).** Audit pending: same pattern may have no-op'd late
  Run A resume sessions after commit f52396f (2026-10-12) — check
  `runs/phase10_text_sft/metrics.jsonl` on Drive for flat loss after a session start.
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
