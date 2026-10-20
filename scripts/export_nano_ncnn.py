# -*- coding: utf-8 -*-
"""导出 ConvNeXt-Nano 三分类模型到 NCNN。

默认读取仓库内 weights/nano_rot90_best_test.pt。
输出目录包含 convnext_nano.param/bin（NCNN 输入 in0、输出 out0）。
"""
import argparse
import shutil
from pathlib import Path

import pnnx
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCH = "convnext_nano.in12k_ft_in1k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=Path, default=ROOT / "weights" / "nano_rot90_best_test.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "build_nano_ncnn")
    ap.add_argument("--imgsz", type=int, default=384)
    args = ap.parse_args()

    if not args.weight.is_file():
        raise SystemExit(f"找不到权重: {args.weight}")
    args.out.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(ARCH, pretrained=False, num_classes=3)
    state = torch.load(args.weight, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    pnnx.export(model, str(args.out / "convnext_nano.pt"), (dummy,))

    param = args.out / "convnext_nano.ncnn.param"
    binary = args.out / "convnext_nano.ncnn.bin"
    if param.exists():
        shutil.move(param, args.out / "convnext_nano.param")
    if binary.exists():
        shutil.move(binary, args.out / "convnext_nano.bin")

    param = args.out / "convnext_nano.param"
    binary = args.out / "convnext_nano.bin"
    if not param.is_file() or not binary.is_file():
        raise SystemExit(f"导出失败，输出目录缺少 param/bin: {args.out}")
    print(f"param={param.stat().st_size} bin={binary.stat().st_size}")
    lines = param.read_text(encoding="utf-8").splitlines()
    print("input:", lines[2])
    print("output:", lines[-1])


if __name__ == "__main__":
    main()
