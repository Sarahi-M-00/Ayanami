#!/usr/bin/env python3
"""Merge a LoRA adapter into base weights WITHOUT loading the model.

Streams base tensors one at a time (safe_open), patches only the targeted
projections with W + (B@A)*(alpha/r) in float32, and writes a new
safetensors file. Base stays untouched. Verifies the result by header
comparison plus a recomputed spot-check.

Usage:
  .venv/bin/python scripts/merge_lora.py \
      --base student_base \
      --adapter runs/smoke_test_focus/checkpoints/checkpoint-24/adapter \
      --out exports/merged_smoke
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja", "generation_config.json")


def read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(args.base)
    adapter = Path(args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    scaling = alpha / r
    print(f"r={r} alpha={alpha} scaling={scaling} "
          f"targets={acfg['target_modules']}", flush=True)

    ahead = read_header(adapter / "adapter_model.safetensors")
    lora_a: dict[str, str] = {}
    for key in ahead:
        if key == "__metadata__" or not key.endswith(".lora_A.weight"):
            continue
        stem = key[len("base_model.model."):-len(".lora_A.weight")]
        lora_a[stem] = key
    print(f"loRA pairs found: {len(lora_a)}", flush=True)
    assert lora_a, "no LoRA weights in adapter"

    merged: dict[str, torch.Tensor] = {}
    patched = 0
    with safe_open(str(adapter / "adapter_model.safetensors"),
                   framework="pt", device="cpu") as fa:
        lora_cache = {stem: (fa.get_tensor(k).to(torch.float32),
                             fa.get_tensor(k[:-len(".lora_A.weight")] +
                                           ".lora_B.weight").to(torch.float32))
                      for stem, k in lora_a.items()}
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as fb:
        for i, name in enumerate(fb.keys()):
            t = fb.get_tensor(name)
            if name[:-len(".weight")] in lora_cache:
                a, b = lora_cache[name[:-len(".weight")]]
                delta = (b @ a) * scaling
                assert delta.shape == tuple(t.shape), (name, delta.shape, t.shape)
                merged[name] = (t.to(torch.float32) + delta).to(t.dtype)
                patched += 1
            else:
                merged[name] = t
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1} tensors", flush=True)

    save_file(merged, out / "model.safetensors")
    for name in COPY_FILES:
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)
    # Copies inherit the read-only snapshot bits; our exports stay writable.
    for p in [out / "model.safetensors"] + [out / n for n in COPY_FILES
                                            if (out / n).is_file()]:
        p.chmod(0o644)

    # ---- verification ----
    check = read_header(out / "model.safetensors")
    check_tensors = {k: v for k, v in check.items() if k != "__metadata__"}
    base_tensors = {k: v for k, v in read_header(base / "model.safetensors").items()
                    if k != "__metadata__"}
    assert set(check_tensors) == set(base_tensors), "tensor name mismatch"
    for k in check_tensors:
        assert check_tensors[k]["shape"] == base_tensors[k]["shape"], k
        assert check_tensors[k]["dtype"] == base_tensors[k]["dtype"], k

    # spot-check: recompute one patched tensor from base + adapter
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as fb, \
            safe_open(str(out / "model.safetensors"), framework="pt", device="cpu") as fm:
        spot = next(iter(lora_cache))
        expect = (fb.get_tensor(spot + ".weight").to(torch.float32)
                  + (lora_cache[spot][1] @ lora_cache[spot][0]) * scaling)
        got = fm.get_tensor(spot + ".weight").to(torch.float32)
        maxdiff = (expect - got).abs().max().item()
        absmax = expect.abs().max().item()
    print(f"tensors={len(check_tensors)} patched={patched} "
          f"spot_maxdiff={maxdiff:.3g} absmax={absmax:.3g}", flush=True)
    # Save/load rounds through bf16 (~3 decimals); allow quantization noise.
    assert maxdiff < 0.01 * max(1.0, absmax), "spot check failed"
    report = {"base": str(base), "adapter": str(adapter), "out": str(out),
              "r": r, "alpha": alpha, "scaling": scaling,
              "tensors": len(check_tensors), "patched": patched,
              "spot_maxdiff": maxdiff,
              "size_bytes": (out / "model.safetensors").stat().st_size}
    (out / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(f"merged -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
