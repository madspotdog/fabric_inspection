# -*- coding: utf-8 -*-
"""Train the two lightweight YOLOv8 classification models."""
import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from rotation_aug import OrthogonalRotation


class RotationTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            dataset.torch_transforms.transforms.insert(0, OrthogonalRotation())
        return dataset


def train_one(name, data, model, args, **kwargs):
    print(f"[{name}] data={data} imgsz={args.imgsz} epochs={args.epochs}")
    net = YOLO(model)
    settings = dict(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, workers=args.workers, seed=42, deterministic=True,
        # Training-time rotation augmentation only; deployment remains no-TTA.
        degrees=90.0, fliplr=0.5, flipud=0.5, scale=0.35, translate=0.08,
        hsv_h=0.015, hsv_s=0.35, hsv_v=0.25, erasing=0.2,
        project=str(args.project), name=name, exist_ok=True,
    )
    settings.update(kwargs)
    net.train(trainer=RotationTrainer, **settings)
    print(f"[{name}] best={args.project / name / 'weights' / 'best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dual_data"))
    ap.add_argument("--project", type=Path, default=Path("dual_runs"))
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--cotton-imgsz", type=int, default=None)
    ap.add_argument("--fiber-imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cotton-model", default=None)
    ap.add_argument("--fiber-model", default=None)
    ap.add_argument("--fiber-data", type=Path, default=None)
    ap.add_argument("--fiber-name", default="pure_fiber6")
    ap.add_argument("--cotton-name", default="cotton_tier")
    ap.add_argument("--skip-fiber", action="store_true")
    ap.add_argument("--skip-cotton", action="store_true")
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"找不到数据集: {args.data}; 先运行 build_dual_datasets.py")
    cotton_model = args.cotton_model or args.model
    fiber_model = args.fiber_model or args.model
    base_imgsz = args.imgsz
    args.imgsz = args.cotton_imgsz or base_imgsz
    if not args.skip_cotton:
        train_one(args.cotton_name, args.data / "cotton_tier", cotton_model, args)
    args.imgsz = args.fiber_imgsz or base_imgsz
    fiber_data = args.fiber_data or (args.data / "pure_fiber6")
    if not args.skip_fiber:
        train_one(args.fiber_name, fiber_data, fiber_model, args, degrees=90.0)


if __name__ == "__main__":
    main()
