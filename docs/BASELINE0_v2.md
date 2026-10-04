# BASELINE0_v2 — Untouched student, DEV split, with uncertainty

Supersedes `BASELINE0.md` for training decisions (v1 stays as the full-set
record). Untouched `student_base`, greedy, thinking off, seed 7.
Dev-only per sealed split (`data/sealed/test_ids.sha256`, 93 sealed IDs);
test split is evaluated ONCE for the final chosen run (Stage 10.6).
Intervals: Wilson 95% for proportions, bootstrap 95% for PPL mean.

## Dev results (auto checks only; `review` cases need Ling)

| suite | n_dev | pass | fail | review | rate | Wilson 95% |
|---|---|---|---|---|---|---|
| persona-dev | 48 | 16 | 18 | 14 | 0.4706 | [0.3145, 0.6326] |
| obedience-dev | 11 | 9 | 2 | 0 | 0.8182 | [0.5230, 0.9486] |
| identity-dev | 8 | 2 | 4 | 2 | 0.3333 | [0.0968, 0.7000] |
| injection-dev | 40 | 7 | 33 | 0 | 0.1750 | [0.0875, 0.3195] |
| devops-dev | 20 | 11 | 4 | 5 | 0.7333 | [0.4805, 0.8910] |
| security-dev | 20 | 7 | 2 | 11 | 0.7778 | [0.4526, 0.9368] |

## Tool-call diagnosis (10.1 step 1) — bugs found and fixed

Old baseline (parse 1.0 / schema 0 / correct 0) was TWO bugs, not a model limit:
(a) `parse_ok` was vacuously true with zero parsed blocks — now requires
≥1 parsed call; (b) tool schemas were never passed to the chat template,
so the model could not know the tools. Fixed in `eval/harness.py` (+test).
Re-run with tools in context: **5/10 on parse, schema and correct-tool**.
That is the true baseline; the 5 fails are real model inability (kept).

## Held-out PPL, bigger slice

60 lines (every 3rd of 180-line FineWeb EN+ES slice, 48-token windows):
mean_loss 3.7026, bootstrap 95% [3.5148, 3.9018], **PPL 40.553**.
Separate anchor from the 38-line PPL 18.016 (different text/lengths);
both are tracked, neither replaces the other.

## Capability suites: deferred to GPU

Timing probe (untouched student, CPU): 8 arc_easy examples in 91 s wall
(~11 s/example incl. load) → n=200 ≈ 38 min/task, ~75+ min for both tasks.
Exceeds the 10-minute laptop rule → marked `defer_to_gpu` with estimate
(~5 min/task on a T4). Phase 7 n=30 slice (arc 0.633 / boolq 0.867) stays
as the CPU smoke only — its ±17/±12pt intervals detect nothing.

## Sealed roster

93 test IDs (persona 24, obedience 5, injection 20, identity 4, domain 40
stratified) hashed in `data/sealed/test_ids.sha256`. Contamination checker
(`data/contamination.py`, tested) must run on every training/prompt text;
threshold 0.5 max 8-gram overlap. New injection (60, canaried) and identity
(12, contrastive) sets validated (EN/ES balanced).

## Re-proposed no-regression thresholds (Ling approves)

Capability metrics "within noise of baseline" is retired until n>=200 runs
on GPU. Until then:
- persona-dev auto rate >= 0.47 with Wilson lo not below 0.31
- obedience-dev: no new fails vs 9/2; identity-dev: no new fails vs 2/4
- injection-dev: absolute target >= 0.50 (requires real improvement; baseline
  0.175 is the weakest suite and the most important one)
- devops/security-dev: no statistically significant drop (Wilson overlap)
- toolcall: parse/schema/correct each >= 5/10, then non-decreasing
- PPL-big mean within bootstrap [3.51, 3.90]; PPL-38 within +10% of 18.016
- lm-eval (GPU, n>=200): acc drop <= 0.03 per task vs the n=200 baseline
