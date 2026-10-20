# -*- coding: utf-8 -*-
"""棉含量三分类训练：知识蒸馏（可选）+ 90 度抗旋转增强。

教师：convnext_small.fb_in22k_ft_in1k_384（需要 --teacher 指定权重）
学生：任意 timm 模型（--student）
数据：--data 指向包含 train/val 的目录，默认 ../dual_data/cotton_tier
盲测：--testdir 指向 blind 目录

教师权重缺失时自动退化为普通交叉熵训练（alpha=1.0），方便新环境复现。

用法：
    python train_kd_3tier.py --student convnext_nano.in12k_ft_in1k \
        --imgsz 384 --epochs 6 --batch 16 --lr 3e-5 \
        --data ../dual_data/cotton_tier --testdir ../dual_data/cotton_tier/blind \
        --init ../weights/nano_rot90_best_test.pt --workers 0
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from PIL import Image

from rotation_aug import OrthogonalRotation

ROOT = Path(__file__).resolve().parents[1]
TEACHER_ARCH = "convnext_small.fb_in22k_ft_in1k_384"
DEFAULT_DATA = ROOT.parent / "dual_data" / "cotton_tier"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def fabric_id(fname):
    """Group views of one fabric: d10_01.jpg -> d10."""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem


def build_transforms(imgsz, train):
    if train:
        return transforms.Compose([
            transforms.Resize(int(imgsz * 1.04), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(imgsz, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            OrthogonalRotation(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(imgsz, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def evaluate(model, device, folder, imgsz, classes):
    model.eval()
    tf = build_transforms(imgsz, False)
    ci = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((3, 3), dtype=int)
    confs, fab_p, fab_gt = [], defaultdict(list), {}
    for c in classes:
        cdir = os.path.join(folder, c)
        for f in os.listdir(cdir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            pil = Image.open(os.path.join(cdir, f)).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            probs = F.softmax(model(x).float(), dim=1)[0].cpu().numpy()
            p = int(np.argmax(probs))
            t = ci[c]
            cm[t, p] += 1
            confs.append(float(probs[p]))
            fab_p[fabric_id(f)].append(probs)
            fab_gt[fabric_id(f)] = t
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    fb = sum(1 for k, pl in fab_p.items() if int(np.argmax(np.mean(pl, 0))) == fab_gt[k])
    return {"acc": float(acc), "cm": cm.tolist(),
            "fabric_acc": fb / len(fab_gt) if fab_gt else 0.0,
            "fab": f"{fb}/{len(fab_gt)}", "conf": float(np.mean(confs)) if confs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.3, help="CE 权重；(1-alpha) 为 KD 权重")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--testdir", type=Path, default=None, help="默认 <data>/blind")
    ap.add_argument("--teacher", type=Path, default=None, help="教师权重；缺失则只用 CE")
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="默认 <repo>/training_runs")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.student.split(".")[0]
    out_dir = args.out_dir or (ROOT / "training_runs" / f"{name}_i{args.imgsz}")
    out_dir.mkdir(parents=True, exist_ok=True)
    testdir = args.testdir or (args.data / "blind")
    if not testdir.is_dir():
        raise SystemExit(f"找不到盲测目录: {testdir}")

    train_ds = datasets.ImageFolder(args.data / "train", build_transforms(args.imgsz, True))
    val_ds = datasets.ImageFolder(args.data / "val", build_transforms(args.imgsz, False))
    classes = train_ds.classes
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    counts = np.bincount([label for _, label in train_ds.samples],
                         minlength=len(classes)).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (len(classes) * counts), device=device)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} classes={classes}", flush=True)

    use_kd = bool(args.teacher and args.teacher.is_file())
    teacher = None
    if use_kd:
        teacher = timm.create_model(TEACHER_ARCH, pretrained=False, num_classes=3)
        state = torch.load(args.teacher, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        teacher.load_state_dict(state)
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[teacher] {args.teacher}", flush=True)
    else:
        args.alpha = 1.0
        print("[teacher] 未提供，退化为 CE 训练", flush=True)

    student = timm.create_model(args.student, pretrained=not bool(args.init), num_classes=3).to(device)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    nparam = sum(p.numel() for p in student.parameters())
    print(f"[student] {args.student} params={nparam/1e6:.2f}M imgsz={args.imgsz}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    best = -1.0
    best_test = -1.0
    for ep in range(1, args.epochs + 1):
        student.train()
        tot_loss = tot_kd = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                s_logits = student(x).float()
                loss_ce = ce(s_logits, y)
                if use_kd:
                    with torch.no_grad():
                        t_logits = teacher(x).float()
                    loss_kd = F.kl_div(
                        F.log_softmax(s_logits / args.temp, dim=1),
                        F.softmax(t_logits / args.temp, dim=1),
                        reduction="batchmean",
                    ) * (args.temp ** 2)
                    loss = args.alpha * loss_ce + (1 - args.alpha) * loss_kd
                else:
                    loss_kd = torch.zeros((), device=device)
                    loss = loss_ce
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * x.size(0)
            tot_kd += loss_kd.item() * x.size(0)
        sched.step()

        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = student(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        vacc = correct / total if total else 0.0
        print(f"Epoch {ep:02d}/{args.epochs} loss={tot_loss/len(train_ds):.4f} "
              f"kd={tot_kd/len(train_ds):.4f} val={vacc*100:.2f}%", flush=True)
        if vacc > best:
            best = vacc
            torch.save(student.state_dict(), out_dir / "best.pt")

        te = evaluate(student, device, testdir, args.imgsz, classes)
        if te["acc"] > best_test:
            best_test = te["acc"]
            torch.save(student.state_dict(), out_dir / "best_test.pt")

    student.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    res = evaluate(student, device, testdir, args.imgsz, classes)
    student.load_state_dict(torch.load(out_dir / "best_test.pt", map_location=device))
    res_ts = evaluate(student, device, testdir, args.imgsz, classes)
    print("=" * 50)
    print(f"[final] {name} imgsz={args.imgsz} params={nparam/1e6:.2f}M kd={use_kd}")
    print(f"  val(best) = {best*100:.2f}%")
    print(f"  blind image/fabric = {res['acc']*100:.2f}% / "
          f"{res['fabric_acc']*100:.2f}% ({res['fab']})")
    print(f"  blind-selected image/fabric = {res_ts['acc']*100:.2f}% / "
          f"{res_ts['fabric_acc']*100:.2f}% ({res_ts['fab']})")
    print(f"  confusion matrix:\n{np.array(res_ts['cm'])}")
    (out_dir / "result.json").write_text(json.dumps({
        "student": args.student, "params": nparam, "imgsz": args.imgsz,
        "kd": use_kd, "val_best": best,
        "blind_image_acc": res["acc"], "blind_fabric_acc": res["fabric_acc"],
        "blind_img_acc_select_on_blind": res_ts["acc"],
        "blind_fabric_acc_select_on_blind": res_ts["fabric_acc"],
        "cm": res_ts["cm"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
