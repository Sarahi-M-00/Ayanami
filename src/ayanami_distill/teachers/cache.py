"""Teacher cache helpers (Phase 4).

Teachers are often too large for 16 GB, so they are never kept resident
during student training. Outputs are generated/scored offline and stored
under data/teacher_cache/<teacher_id>/<revision>/. Logits are stored
top-k only: int32 ids + float16 logprobs + one float16 tail mass.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TOPK = 32
BYTES_ID = 4       # int32 token id
BYTES_LOGPROB = 2  # float16 logprob


def cache_dir(root: Path, teacher_id: str, revision: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in teacher_id)
    return Path(root) / "data" / "teacher_cache" / safe / revision


def estimate_topk_cache(n_tokens: int, k: int = DEFAULT_TOPK) -> dict:
    """Disk estimate for cached top-k logprobs. No files touched."""
    per_token = k * (BYTES_ID + BYTES_LOGPROB) + BYTES_LOGPROB  # ids + logprobs + tail
    total = n_tokens * per_token
    return {"n_tokens": n_tokens, "k": k, "bytes_per_token": per_token,
            "total_bytes": total, "total_mib": round(total / 2**20, 2)}


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
