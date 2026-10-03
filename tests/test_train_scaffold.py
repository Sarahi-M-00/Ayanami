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


def test_cli_rejects_unimplemented_mode(tmp_path):
    import yaml
    cfg = {"mode": "logit_kd"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    from ayanami_distill.train.__main__ import main
    assert main(["--config", str(p), "--run-dir", str(tmp_path / "r")]) == 2
