# -*- coding: utf-8 -*-
"""Evaluate both models and the cross-check rules on grouped blind data."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def cotton_business(name):
    return {"Tier0_Low": "低棉", "低棉": "低棉",
            "Tier1_Mid": "中棉", "混纺": "中棉", "中棉": "中棉",
            "Tier2_High": "高棉", "高棉": "高棉"}.get(name, name)


def fiber_business(name):
    return {"棉": "纯棉", "纯棉": "纯棉", "羊毛": "纯羊毛", "纯羊毛": "纯羊毛",
            "涤纶": "纯涤纶", "纯涤纶": "纯涤纶",
            "丝": "其他", "亚麻": "其他", "其他": "其他"}.get(name, name)


def evaluate(model_path, data, imgsz, business):
    model = YOLO(str(model_path))
    total = correct = 0
    per = {}
    groups = defaultdict(list)
    for cls_dir in sorted((data / "blind").iterdir()):
        if not cls_dir.is_dir():
            continue
        for image in cls_dir.iterdir():
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            result = model.predict(str(image), imgsz=imgsz, verbose=False)[0]
            raw_pred = model.names[int(result.probs.top1)]
            pred = business(raw_pred)
            truth = business(cls_dir.name)
            total += 1
            per.setdefault(truth, [0, 0])[0] += 1
            if pred == truth:
                correct += 1
                per[truth][1] += 1
            groups[(truth, image.stem.rsplit("_", 1)[0])].append(pred)
    fabric_correct = sum(Counter(preds).most_common(1)[0][0] == truth
                         for (truth, _), preds in groups.items())
    return model, correct, total, per, fabric_correct, len(groups)


def evaluate_timm(model_path, data, imgsz, business, tta=True):
    import timm
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("convnext_nano.in12k_ft_in1k", pretrained=False, num_classes=3)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    transform = T.Compose([
        T.Resize(imgsz, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(imgsz), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    total = correct = 0
    per = {}
    groups = defaultdict(list)
    with torch.no_grad():
        for cls_dir in sorted((data / "blind").iterdir()):
            if not cls_dir.is_dir():
                continue
            truth = business(cls_dir.name)
            for image in cls_dir.iterdir():
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                pil = Image.open(image).convert("RGB")
                angles = (0, 90, 180, 270) if tta else (0,)
                batch = torch.stack([transform(TF.rotate(pil, angle)) for angle in angles]).to(device)
                probs = torch.softmax(model(batch).float(), dim=1).mean(0).cpu().numpy()
                pred = ("低棉", "中棉", "高棉")[int(np.argmax(probs))]
                total += 1
                per.setdefault(truth, [0, 0])[0] += 1
                if pred == truth:
                    correct += 1
                    per[truth][1] += 1
                groups[(truth, image.stem.rsplit("_", 1)[0])].append(probs)
    fabric_correct = sum(int(np.argmax(np.mean(probs, axis=0))) == ("低棉", "中棉", "高棉").index(truth)
                         for (truth, _), probs in groups.items())
    return model, correct, total, per, fabric_correct, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "dual_data")
    ap.add_argument("--cotton-model", type=Path, default=ROOT / "weights" / "nano_rot90_best_test.pt")
    ap.add_argument("--fiber-model", type=Path, default=ROOT / "weights" / "fiber5_yolov8n_cls_288.pt")
    ap.add_argument("--imgsz", type=int, default=None, help="同时覆盖两个输入尺寸")
    ap.add_argument("--cotton-imgsz", type=int, default=384)
    ap.add_argument("--fiber-imgsz", type=int, default=288)
    args = ap.parse_args()
    if args.imgsz is not None:
        args.cotton_imgsz = args.fiber_imgsz = args.imgsz
    for path in (args.cotton_model, args.fiber_model):
        if not path.exists():
            raise SystemExit(f"找不到模型: {path}")
    fiber_data = args.data / "pure_fiber6"
    if not fiber_data.exists():
        fiber_data = args.data / "pure_fiber"
    for title, model_path, data, business, imgsz in (("棉含量", args.cotton_model, args.data / "cotton_tier", cotton_business, args.cotton_imgsz),
                                                      ("纯面料", args.fiber_model, fiber_data, fiber_business, args.fiber_imgsz)):
        name = str(model_path).lower()
        is_timm = title == "棉含量" and any(tag in name for tag in ("nano", "rot90", "convnext"))
        if is_timm:
            result = evaluate_timm(model_path, data, imgsz, business, tta=True)
        else:
            result = evaluate(model_path, data, imgsz, business)
        _, ok, total, per, fabric_ok, fabric_total = result
        print(f"{title}: image={ok}/{total} = {100 * ok / total:.2f}%  "
              f"group={fabric_ok}/{fabric_total} = {100 * fabric_ok / fabric_total:.2f}%")
        print("  " + " ".join(f"{k}:{v[1]}/{v[0]}" for k, v in per.items() if v[0]))
    print("交叉规则: 低棉+其他、或高棉+纯棉视为通过；其余组合标记为复核。")


if __name__ == "__main__":
    main()
