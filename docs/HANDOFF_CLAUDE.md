# HANDOFF FOR CLAUDE (or any successor agent) — ayanami-distill, Stage 10.3 done

Copy everything below the line into a new Claude conversation. Attach or link
https://github.com/Sarahi-M-00/Ayanami if you can. Ask Claude to act as
co-engineer/reviewer from Stage 10.3 onward.

---

You are continuing the **ayanami-distill** project (repo above, branch `main`).
Mission: build **Ayanami**, a small (1.7B), local, loyal AI assistant created
by **Ling**, expert in **defensive cybersecurity + DevOps**, that learns from
teacher models via distillation. Student: `orlandorubino/Qwen3-1.7B-heretic`
(Qwen3-1.7B abliterated/BF16, no refusals left).

HARD RULES (non-negotiable, from the project prompts):
- Work phase by phase; each stage ends at a STOP GATE needing Ling's "continue".
- Never invent values: read them from repo files. Anything unverified is marked as such.
- Student weights (`student_base/`) are READ-ONLY. Never modify them.
- Pin everything (SHAs, versions, seeds). Log non-trivial decisions in `docs/DECISIONS.md`.
- Defensive + authorized-scope security ONLY. No offensive-capability work, ever.
- No secrets in files. Everything written (code/docs) in English.
- Laptop (HP EliteBook, 6 cores, 14.9 GB RAM, no GPU) is `cpu_only`: never run
  jobs >~10 min on it. Heavy work runs on remote GPU (Kaggle/Colab) via LING ACTIONs
  (steps Ling executes); prepare ready-to-run packages, then wait. Never pretend remote work happened.
- Sealed test IDs (`data/sealed/test_ids.sha256`, 93 hashes) are evaluated ONCE, only for the final run.

STATE (all verified, evidence in repo):
- Phases 0–9 + 10.0–10.3 done, 55/55 tests green. No real distillation has run yet.
- Student pinned: HF rev `433869992d…`, 1,720,574,976 params BF16, tokenizer
  fingerprint `563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa`.
- Baselines (untouched, greedy, thinking OFF, seed 7): persona 25/24/23 (72);
  obedience .75; devops .77; security .57; injection 2/8/2 (now 60 canaried cases);
  toolcall TRUE baseline 5/10 (an old 1.0/0/0 was two harness bugs, fixed+tested);
  PPL 18.016 (38-line) and 40.553 (60-line strided FineWeb slice);
  arc_easy 0.633 / boolq 0.867 at n=30 (too noisy; n>=200 deferred to GPU).
- Dev/test sealed splits done (dev rerun complete with Wilson intervals in
  `docs/BASELINE0_v2.md`); prompt pool v1 = 1,911 prompts (deduped, sealed-clean,
  sources in `docs/DATA_SOURCES.md`, stats in `docs/DATA_STATS.md`).
- Teacher APPROVED by Ling: `Qwen/Qwen3-8B` (Instruct), Apache-2.0, tokenizer
  fingerprint byte-identical to student (logit_kd possible). Thinking OFF.
  Non-thinking sampling per its card: T=0.7/p=0.8/k=20/minP=0. Rejected: Pingu
  Unchained (commercial API, offensive/poisoned, ToS unverified).
- Compute: Kaggle first (~30 GPU-h/week floating, verify in-account), Colab T4
  overflow, single GPU, fp16 (T4/P100 lack bf16 — verified). GPU ledger empty.
- Export loop proven on a smoke adapter: merge→GGUF→Q4_K_M preserves behavior
  (Q4 17.8 tok/s); `llama-cli` of this vintage needs `-st`.
- Thinking OFF by default everywhere (recorded decision).

KEY FILES: `docs/PROJECT_REPORT.md` (all gates), `docs/PHASE10_DECISIONS.md`
(all 6 items confirmed by Ling), `docs/BASELINE0_v2.md` (thresholds proposed),
`docs/RUNBOOK.md` (incl. "Remote teacher run"), `scripts/teacher_cache.py` +
`notebooks/teacher_cache.ipynb` (pilot-ready), `data/processed/pilot_50.jsonl`
(50 stratified prompts), `configs/` (pins, LoRA, distill modes, scope.yaml).

WHAT IS NEXT (in order):
1. LING ACTION (human does this): run the 50-prompt pilot on Kaggle per
   RUNBOOK ("Remote teacher run"), upload `data/teacher_cache/qwen-qwen3-8b/`,
   record hours in `docs/GPU_LEDGER.md`. Then tell the agent.
2. Agent ingests/validates shards, extrapolates GPU hours, eye-reviews records.
3. Stage 10.4: full cache → 10.5: train Run A (text_sft) + Run B (logit_kd) on
   remote GPU → 10.6: eval vs baseline v2, winner by pre-approved thresholds,
   sealed test once → 10.7: export Q4 → 10.8: handoff.
4. Open items needing Ling: 7 thresholds approval, 80 domain reference review,
   scope.yaml real assets, GPU hours logging.

Verify before trusting: `git log --oneline -5`, `pytest tests/ -q` (expect 55
passed), `cat docs/GPU_LEDGER.md`. Question anything that disagrees with files.
