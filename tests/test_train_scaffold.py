"""Phase 5 scaffold tests: mix builder, checkpoint IO, on-policy stub, CLI parse."""

import json

import torch

from ayanami_distill.data.mix import build_mix
from ayanami_distill.train import on_policy as OP
from ayanami_distill.train.__main__ import parse_args
from ayanami_distill.train.trainer import load_training_state, save_training_state


def test_build_mix_proportions_and_determinism():
    a = [{"i": i} for i in range(100)]
    b = [{"j": i} for i in range(100)]
    items, prov = build_mix({"big": (a, 0.7), "small": (b, 0.3)}, 1000, seed=7)
    assert len(items) == 1000 and sum(prov["counts"].values()) == 1000
    assert abs(prov["ratios"]["big"] - 0.7) < 0.05
    items2, _ = build_mix({"big": (a, 0.7), "small": (b, 0.3)}, 1000, seed=7)
    assert items == items2
    items3, _ = build_mix({"big": (a, 0.7), "small": (b, 0.3)}, 1000, seed=8)
    assert items != items3


def test_build_mix_rejects_empty_pool_and_bad_weights():
    import pytest
    with pytest.raises(ValueError):
        build_mix({"x": ([], 1.0)}, 5, seed=1)
    with pytest.raises(ValueError):
        build_mix({"x": ([{"a": 1}], 0.0)}, 5, seed=1)


def test_checkpoint_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    x = torch.randn(3, 4)
    opt.zero_grad()
    layer(x).sum().backward()
    opt.step()
    sched.step()
    before = [p.detach().clone() for p in layer.parameters()]
    save_training_state(tmp_path / "state.pt", opt, sched, 11)
    # Perturb, then restore.
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(1.0)
    opt2 = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 0.5)
    step = load_training_state(tmp_path / "state.pt", opt2, sched2)
    assert step == 11
    assert list(opt2.state_dict()["state"].keys()) == list(opt.state_dict()["state"].keys())
    assert before  # params themselves live in the model/adapter checkpoint


def test_on_policy_is_interface_only():
    import pytest
    with pytest.raises(NotImplementedError):
        OP.on_policy_step(None, None, [])


def test_cli_parse_defaults():
    args = parse_args(["--config", "configs/distill/text_sft.yaml"])
    assert args.config == "configs/distill/text_sft.yaml" and args.run_dir is None
    args2 = parse_args(["--config", "x.yaml", "--run-dir", "runs/r1"])
    assert args2.run_dir == "runs/r1"


def test_cli_rejects_unknown_mode(tmp_path):
    import yaml
    cfg = {"mode": "hidden_kd"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    from ayanami_distill.train.__main__ import main
    assert main(["--config", str(p), "--run-dir", str(tmp_path / "r")]) == 2


# ---- Phase 10.5: record -> SFT/KD encoding ----
class _StubTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 300 for c in text]}


def _kd_rec():
    return {
        "id": "t-1",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "hi"}],
        "student_messages": [{"role": "user", "content": "hi"}],
        "completion": "yo",
    }


def test_sft_messages_appends_completion():
    from ayanami_distill.train.trainer import sft_messages_from_record
    msgs = sft_messages_from_record(_kd_rec())
    assert msgs[-1] == {"role": "assistant", "content": "yo"}
    assert msgs[0]["role"] == "user"


def test_sft_messages_keeps_existing_assistant():
    from ayanami_distill.train.trainer import sft_messages_from_record
    rec = {"id": "t-2", "messages": [{"role": "user", "content": "hi"},
                                     {"role": "assistant", "content": "hey"}]}
    assert sft_messages_from_record(rec) == rec["messages"]


def test_sft_messages_rejects_empty_completion():
    import pytest
    from ayanami_distill.train.trainer import sft_messages_from_record
    with pytest.raises(ValueError):
        sft_messages_from_record({"id": "t-3", "messages": [{"role": "user", "content": "hi"}],
                                        "completion": "  "})


def test_encode_kd_example_aligns_positions():
    from ayanami_distill.losses.text_sft import assistant_mask_from_messages
    from ayanami_distill.train.trainer import encode_kd_example
    tok, rec, k = _StubTok(), _kd_rec(), 4
    # KD aligns on the FULL teacher context (rec["messages"] + completion).
    msgs = list(rec["messages"]) + [{"role": "assistant", "content": rec["completion"]}]
    ids, _ = assistant_mask_from_messages(tok, msgs)
    n = len(ids)
    pos = list(range(n - 3, n))  # last 3 tokens are assistant content
    rec["logprobs_topk"] = {"k": k, "positions": pos,
                            "ids": [[1] * k for _ in pos],
                            "logprobs": [[-0.5] * k for _ in pos],
                            "tail_mass": [0.1] * len(pos)}
    ex = encode_kd_example(tok, rec, 2048)
    assert ex is not None and ex["pos"] == pos
    assert len(ex["t_ids"]) == len(pos)


def test_encode_kd_example_skips_bad_shapes():
    from ayanami_distill.train.trainer import encode_kd_example
    tok, rec = _StubTok(), _kd_rec()
    rec["logprobs_topk"] = None
    assert encode_kd_example(tok, rec, 2048) is None
    rec["logprobs_topk"] = {"k": 4, "positions": [10**9], "ids": [[1] * 4],
                            "logprobs": [[-0.5] * 4], "tail_mass": [0.1]}
    assert encode_kd_example(tok, rec, 2048) is None
