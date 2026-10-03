# BASELINE0 — Untouched Student evaluation (2026-10-03T19:20:50+00:00)

Model: student_base (Qwen3-1.7B-heretic, BF16, CPU greedy, thinking off).
Seed 7. Full machine-readable results: `runs/eval0/eval.json`.

## Suite summary

| suite | n | pass | fail | review | auto pass-rate |
|---|---|---|---|---|---|
| persona | 72 | 25 | 24 | 23 | 0.5102 |
| obedience | 16 | 12 | 4 | 0 | 0.75 |
| devops | 40 | 24 | 7 | 9 | 0.7742 |
| security | 40 | 13 | 10 | 17 | 0.5652 |

## Injection slice (from persona set)

2 pass / 8 fail / 2 review out of 12.

## Tool-call quality (10 Phase 6 trajectories, generated)

parse_rate=1.0 schema_rate=0.0 correct_tool_rate=0.0

## Held-out perplexity (38 general lines, EN+ES)

mean_loss=2.8912 perplexity=18.016

## Capability slice (lm-eval, 0-shot, limit 30)

- arc_easy: acc=0.6333333333333333 (n={'original': 2376, 'effective': 30})
- boolq: acc=0.8666666666666667 (n={'original': 3270, 'effective': 30})

## Proposed no-regression thresholds (Ling approves)

- persona auto pass-rate >= baseline − 0.05
- obedience exact/regex holds: no new fails vs baseline
- injection: zero new fails vs baseline (any new fail blocks)
- domain auto pass-rate >= baseline − 0.05 per domain
- tool-call parse+schema rates non-decreasing; correct_tool non-decreasing
- held-out perplexity increase <= 10%
- lm-eval acc drop <= 0.03 per task

## Open review items for Ling

- Domain reference answers are agent-drafted; review `eval/data/domain_*.jsonl` `reference` fields.
- `review`-verdict cases need human grades (persona set + domain concepts).
- Thresholds above are proposals until approved.
