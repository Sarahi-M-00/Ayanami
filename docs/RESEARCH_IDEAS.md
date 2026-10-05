# Ayanami research ideas (parked)

Ideas worth pursuing later, ordered by dependency. None are Phase 10 work.

## R-01 — Cross-tokenizer logit "translator" [PARKED — needs own GPU]

- **Idea (Ling, 2026-10-06):** a logit-KD bridge between models with different
  vocabularies, so we are not limited to same-tokenizer teachers and get
  richer signal than `kd_text`.
- **Known prior art:** cross-tokenizer / Universal Logit Distillation (ULD):
  align both tokenizations over decoded text (edit-distance / DP), project
  teacher distributions onto student vocab via shared-token anchors or
  byte level, train KL on the mapped distribution.
- **Why parked:** approximate mapping injects noise; complex to implement
  well; higher compute cost; current priority is finishing the Qwen bulk and
  the A/B (text_sft vs logit_kd) on borrowed GPU.
- **Revisit when:** (a) own GPU available (investment), AND (b) Run B
  (logit_kd) beats Run A (text_sft) — that result would justify investing in
  better logit transfer. Multi-teacher text KD (kd_text from several
  Apache-2.0 teachers) is the cheaper intermediate step.
- **Constraints on revisit:** teachers must be Apache-2.0/MIT verified
  (license recorded with fingerprint, as with Qwen3-8B); defensive +
  authorized-scope only, unchanged.
