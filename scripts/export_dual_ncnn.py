# -*- coding: utf-8 -*-
"""Export both YOLO classification weights to NCNN for Raspberry Pi."""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_yolo_one(weight, output, imgsz):
    output.mkdir(parents=True, exist_ok=True)
    result = YOLO(str(weight)).export(format="ncnn", imgsz=imgsz, device="cpu")
    source = Path(result)
    param = next(source.glob("*.param"), None)
    binary = next(source.glob("*.bin"), None)
    if not param or not binary:
        raise RuntimeError(f"导出目录没有 NCNN 文件: {source}")
    shutil.copy2(param, output / "model.ncnn.param")
    shutil.copy2(binary, output / "model.ncnn.bin")
    model = YOLO(str(weight))
    names = [model.names[i] for i in range(len(model.names))]
    (output / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{weight} -> {output}")


def copy_nano_model(output, source):
    output.mkdir(parents=True, exist_ok=True)
    param = source / "cotton_tier_nano.ncnn.param"
    binary = source / "cotton_tier_nano.ncnn.bin"
    if not param.is_file() or not binary.is_file():
        raise FileNotFoundError(f"缺少已有 Nano NCNN 文件: {source}")
    shutil.copy2(param, output / "model.ncnn.param")
    shutil.copy2(binary, output / "model.ncnn.bin")
    (output / "labels.json").write_text(
        json.dumps(["Tier0_Low", "Tier1_Mid", "Tier2_High"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{param} -> {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cotton-model", type=Path, default=Path("student_kd/mjc_convnext_nano_i384/best_test.pt"))
    ap.add_argument("--fiber-model", type=Path, default=Path("deliver/fiber5_yolov8n_cls_288.pt"))
    ap.add_argument("--out", type=Path, default=Path("rpi_models"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--cotton-imgsz", type=int, default=384)
    ap.add_argument("--fiber-imgsz", type=int, default=288)
    ap.add_argument("--cotton-nano-source", type=Path, default=Path("nano_ncnn"))
    args = ap.parse_args()
    for weight in (args.cotton_model, args.fiber_model):
        if not weight.exists():
            raise SystemExit(f"找不到权重: {weight}")
    # The Nano precision model is a timm state_dict; its verified PNNX/NCNN
    # export is kept in nano_ncnn. Fiber5 is a standard Ultralytics checkpoint.
    copy_nano_model(args.out / "cotton_tier", args.cotton_nano_source)
    export_yolo_one(args.fiber_model, args.out / "pure_fiber", args.fiber_imgsz)


if __name__ == "__main__":
    main()
