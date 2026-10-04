# DATA_STATS — prompt pool v1 (`data/processed/prompts_v1.jsonl`)

Built 2026-10-04 by `scripts/build_prompts.py`. 1,911 prompts, 0 exact dups,
0 near-dups (Jaccard-5g ≥ 0.85), 0 sealed contaminations, 0 schema-invalid.
All `split=train` in v1 (dev/test assignment happens at dataset build, 10.4).

## Counts by source

| source | n | share |
|---|---|---|
| fineweb EN replay | 400 | 20.9% |
| authored:obedience | 376 | 19.7% |
| authored:injection | 336 | 17.6% |
| authored:domain (devops+security Q&A) | 288 | 15.1% |
| authored:identity (contrastive) | 216 | 11.3% |
| authored:replay-es (hand ES lines) | 100 | 5.2% |
| authored:commands (devops probes) | 100 | 5.2% |
| authored:tool (registry tasks) | 95 | 5.0% |

## Counts by (record category, lang)

Record categories follow the v1 enum (obedience/injection live under
`general`, distinguished by the `source` prefix for mixing).

| (category, lang) | n |
|---|---|
| (general, en) | 756 |
| (general, es) | 456 |
| (persona, en) | 108 |
| (persona, es) | 108 |
| (devops, en) | 128 |
| (devops, es) | 106 |
| (security, en) | 80 |
| (security, es) | 74 |
| (tool_use, en) | 61 |
| (tool_use, es) | 34 |

## User-turn token lengths (student tokenizer)

min 3 / median 13 / p95 146 / max 259. Short prompts: cheap on a T4.

## Student conditioning (Appendix A)

- identity: 50% with persona prompt / 50% bare (verified 0.50)
- obedience/injection/tool/devops/security: with persona prompt
- replay: bare user turn only

## Known deviation from Appendix B (written reason, 10.4 may adjust)

v1 is author-heavy by design (quality-first): tool-use 5.0% vs 18% target
(hand-curated tasks only) and replay ~26% vs 15% (cheap bulk). The pilot
(10.3) fixes the final size; expansion (teacher paraphrase on GPU) rebalances
toward targets in v2, or shares are re-justified. Counts, not verifiers,
are what gets cut first.
