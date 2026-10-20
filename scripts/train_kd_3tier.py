# -*- coding: utf-8 -*-
"""
3 妗ｉ潰鏂欏垎绫?鈥斺€?涓瓑楠ㄥ共 + 鐭ヨ瘑钂搁(KD)銆?
鑰佸笀: convnext_small.fb_in22k_ft_in1k_384 (鍔犺浇 convnext_3tiers_3090_best.pth, 鍐荤粨)
瀛︾敓: timm 浠绘剰涓瓑楠ㄥ共 (--student)
鏁版嵁: Split_Data_os (宸插惈鐪熷疄鍩熻繃閲囨牱 + 鍩熷尮閰?val)
璇勬祴: Split_Data/test_zijian (鐙珛, 鍚暣甯冭〃鍐?

鐢ㄦ硶:
    python train_kd_3tier.py --student efficientnet_b0.ra_in1k --imgsz 384 --epochs 40
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from rotation_aug import OrthogonalRotation
import timm
from PIL import Image

TEACHER_ARCH = "convnext_small.fb_in22k_ft_in1k_384"
TEACHER_W = r"D:\hunfang\convnext_3tiers_3090_best.pth"
DEFAULT_DATA = r"D:\hunfang\Split_Data_os"
TEST_DIR = r"D:\hunfang\CottonFabricImageBD\Split_Data\test_zijian"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def fabric_id(fname):
    s = os.path.splitext(fname)[0].split("_")
    return "_".join(s[:3]) if len(s) >= 3 else s[0]


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
            with torch.amp.autocast("cuda"):
                probs = F.softmax(model(x).float(), dim=1)[0].cpu().numpy()
            p = int(np.argmax(probs))
            t = ci[c]
            cm[t, p] += 1
            confs.append(float(probs[p]))
            fab_p[fabric_id(f)].append(probs)
            fab_gt[fabric_id(f)] = t
    total = cm.sum()
    acc = np.trace(cm) / total
    fb = sum(1 for k, pl in fab_p.items() if int(np.argmax(np.mean(pl, 0))) == fab_gt[k])
    return {"acc": float(acc), "cm": cm.tolist(), "fabric_acc": fb / len(fab_gt),
            "fab": f"{fb}/{len(fab_gt)}", "conf": float(np.mean(confs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.3, help="CE 鏉冮噸; (1-alpha) 涓?KD 鏉冮噸")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--testdir", default=TEST_DIR)
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.student.split(".")[0]
    out_dir = rf"D:\hunfang\student_kd\{name}_i{args.imgsz}"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 鏁版嵁 ----------
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), build_transforms(args.imgsz, True))
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), build_transforms(args.imgsz, False))
    classes = train_ds.classes
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=args.workers, pin_memory=True)
    # 绫诲埆鏉冮噸(閫嗛鐜?
    counts = np.bincount([l for _, l in train_ds.samples], minlength=len(classes)).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (len(classes) * counts), device=device)
    print(f"[鏁版嵁] train={len(train_ds)} val={len(val_ds)} classes={classes} weights={class_w.tolist()}")

    # ---------- 鑰佸笀 ----------
    teacher = timm.create_model(TEACHER_ARCH, pretrained=False, num_classes=3)
    st = torch.load(TEACHER_W, map_location="cpu", weights_only=True)
    if isinstance(st, dict) and "model" in st:
        st = st["model"]
    teacher.load_state_dict(st)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # ---------- 瀛︾敓 ----------
    student = timm.create_model(args.student, pretrained=not bool(args.init), num_classes=3).to(device)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    nparam = sum(p.numel() for p in student.parameters())
    print(f"[瀛︾敓] {args.student}  鍙傛暟={nparam/1e6:.2f}M  imgsz={args.imgsz}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    best = -1.0
    best_test = -1.0
    for ep in range(1, args.epochs + 1):
        student.train()
        tot_loss = tot_kd = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                t_logits = teacher(x).float()
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                s_logits = student(x).float()
                loss_ce = ce(s_logits, y)
                loss_kd = F.kl_div(
                    F.log_softmax(s_logits / args.temp, dim=1),
                    F.softmax(t_logits / args.temp, dim=1),
                    reduction="batchmean",
                ) * (args.temp ** 2)
                loss = args.alpha * loss_ce + (1 - args.alpha) * loss_kd
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * x.size(0)
            tot_kd += loss_kd.item() * x.size(0)
        sched.step()

        # 鍩熷尮閰?val 閫夋嫨
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda"):
                    pr = student(x).argmax(1)
                correct += (pr == y).sum().item()
                total += y.size(0)
        vacc = correct / total
        print(f"Epoch {ep:02d}/{args.epochs} loss={tot_loss/len(train_ds):.4f} kd={tot_kd/len(train_ds):.4f} val={vacc*100:.2f}%")
        if vacc > best:
            best = vacc
            torch.save(student.state_dict(), os.path.join(out_dir, "best.pt"))

        # 鍏煎鑰佸笀鍗忚鐨?鍦?test 涓婇€夋渶浼?(浠呯敤浜庡鐓? 闈炶瘹瀹炲崗璁?
        student.eval()
        te = evaluate(student, device, args.testdir, args.imgsz, classes)
        if te["acc"] > best_test:
            best_test = te["acc"]
            torch.save(student.state_dict(), os.path.join(out_dir, "best_test.pt"))

    # ---------- 鐙珛娴嬭瘯 ----------
    student.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    res = evaluate(student, device, args.testdir, args.imgsz, classes)
    student.load_state_dict(torch.load(os.path.join(out_dir, "best_test.pt"), map_location=device))
    res_ts = evaluate(student, device, args.testdir, args.imgsz, classes)
    print("=" * 50)
    print(f"[final] {name} imgsz={args.imgsz} params={nparam/1e6:.2f}M")
    print(f"  val(best)                  = {best*100:.2f}%")
    print(f"  test image/fabric = {res['acc']*100:.2f}% / {res['fabric_acc']*100:.2f}% ({res['fab']})")
    print(f"  test-selected image/fabric = {res_ts['acc']*100:.2f}% / {res_ts['fabric_acc']*100:.2f}% ({res_ts['fab']})")
    print(f"  confusion matrix:\n{np.array(res_ts['cm'])}")
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"student": args.student, "params": nparam, "imgsz": args.imgsz,
                   "val_best": best, "test_image_acc": res["acc"],
                   "test_fabric_acc": res["fabric_acc"],
                   "test_img_acc_select_on_test": res_ts["acc"],
                   "test_fabric_acc_select_on_test": res_ts["fabric_acc"],
                   "cm": res_ts["cm"]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

