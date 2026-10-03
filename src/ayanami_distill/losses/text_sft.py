"""text_sft: next-token cross-entropy masked to assistant tokens only.

System, user and tool-output tokens never receive loss. The mask is built
from cumulative chat-template renders: each message block starts with the
atomic <|im_start|> token, so tokenizing a prefix is a strict prefix of
tokenizing the longer render. If that invariant ever breaks, we raise
instead of silently training on the wrong tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def assistant_mask_from_messages(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Return (input_ids, loss_mask) with mask=1 on assistant content tokens.

    The assistant turn header (<|im_start|>assistant\\n) is masked out; the
    terminating <|im_end|> is masked in (it teaches stopping). Tool-result
    turns (role "tool") and system/user turns get mask 0.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    mask = [0] * len(full_ids)

    header_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    for j, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prev_text = tokenizer.apply_chat_template(messages[:j], tokenize=False,
                                                  add_generation_prompt=False)
        cur_text = tokenizer.apply_chat_template(messages[:j + 1], tokenize=False,
                                                 add_generation_prompt=False)
        prev_ids = tokenizer(prev_text, add_special_tokens=False)["input_ids"] if prev_text else []
        cur_ids = tokenizer(cur_text, add_special_tokens=False)["input_ids"]
        if cur_ids[:len(prev_ids)] != prev_ids:
            raise ValueError(
                f"prefix instability at assistant turn {j}: refusing to guess the span")
        span = cur_ids[len(prev_ids):]
        if span[:len(header_ids)] != header_ids:
            raise ValueError(f"assistant turn {j} does not start with the expected header")
        start = len(prev_ids) + len(header_ids)
        for t in range(start, len(prev_ids) + len(span)):
            mask[t] = 1

    if len(full_ids) != len(mask):
        raise ValueError("mask length mismatch")
    return full_ids, mask


def build_labels(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), input_ids, torch.full_like(input_ids, IGNORE_INDEX))


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted cross-entropy over masked labels. Scalar; ignores IGNORE_INDEX."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=IGNORE_INDEX)
