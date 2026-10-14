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
- Phase: **10.5** — Run A (text_sft) relay-training on Colab T4 toward step 1000,
  checkpoints every 50 auto-synced to Drive (`AYANAMI_OUTBOX`). Then Run B (logit_kd).
- Last verified: suite 77 green (`pytest tests/ -q`).
- Next: finish A → run B → 10.6 compare + evals → merge/export Q4.

## Hard rules
- Local open teachers only (Apache-2.0/MIT, license + fingerprint recorded).
- Defensive + authorized scope only (`configs/scope.yaml`, default-deny).
- 100% heretic: no refusal training; limits live in code, not weights.
- Rejected teachers (ToS): Grok, Claude — reference only.
- Push needs Ling (SSH passphrase); Colab clones fresh after history rewrites.
