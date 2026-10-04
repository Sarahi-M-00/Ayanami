# DATA_SOURCES — prompt pool v1 (Stage 10.2)

Every prompt source with its license. Rule: anything with unclear terms is
excluded. No teacher outputs were used (laptop stage; generation is 10.3).

| source prefix | origin | license / terms | revision / date |
|---|---|---|---|
| `authored:*` | Hand-written by the agent in `scripts/build_prompts.py` (identity Qs, obedience constraints, injection frames, tool tasks, domain Q&A, ES replay lines, command probes) | Original work for this project (same license as the repo) | repo commit (see git log) |
| `fineweb:HuggingFaceFW/fineweb/sample-100BT` | HuggingFaceFW FineWeb EN sample, streamed (400 docs) | ODC-By 1.0 — attribution required (this file is the attribution record) | dataset rev `9bb295dd…` (Hub API, 2026-10-04) |
| `authored:replay-es` | Hand-written ES general lines in the builder | Original work, as above | repo commit |

Excluded by rule: scraped data, datasets with unclear/non-permissive terms,
any teacher-generated text (nothing of the kind exists yet).
Replay data is general web/hand text only — no eval content. The pool was
contamination-checked against all 93 sealed test IDs (0 overlaps).
