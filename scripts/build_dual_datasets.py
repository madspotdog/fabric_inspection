# -*- coding: utf-8 -*-
"""Build grouped train/val/blind datasets for the two-model fabric pipeline.

The split is made by fabric/sample id, never by individual image.  This is
important because each fabric has several microscope views.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TIERS = ("Tier0_Low", "Tier1_Mid", "Tier2_High")
TIER_BY_RANGE = {"0": "Tier0_Low", "1-50": "Tier0_Low",
                 "50-79": "Tier1_Mid", "80-100": "Tier2_High"}
FIBERS = ("纯棉", "纯羊毛", "纯涤纶", "其他")
RAW_FIBERS = ("丝", "亚麻", "其他", "棉", "涤纶", "羊毛")


def images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def group_id(path: Path) -> str:
    # d10_01 and a1_05 are views of one fabric. External files use a stable
    # filename prefix; retaining the parent folder avoids cross-source clashes.
    stem = path.stem
    # Microscope captures use *_01 ... *_05 for one fabric. Public/external
    # images have no view suffix, so each source image remains its own group.
    parts = stem.split("_")
    if len(parts) >= 2 and re.fullmatch(r"\d{2}", parts[-1]):
        prefix = "_".join(parts[:-1])
    else:
        prefix = stem
    return f"{path.parent.name}_{prefix}"


def clear_make(root: Path, classes):
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "blind"):
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def split_groups(groups, seed, val_ratio=.15, blind_ratio=.20):
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    n_blind = max(1, round(len(keys) * blind_ratio))
    n_val = max(1, round((len(keys) - n_blind) * val_ratio))
    return keys[n_blind + n_val:], keys[n_blind:n_blind + n_val], keys[:n_blind]


def copy_one(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_grouped(records, out, classes, seed):
    clear_make(out, classes)
    by_class = {c: defaultdict(list) for c in classes}
    for cls, src in records:
        by_class[cls][group_id(src)].append(src)
    counts = defaultdict(lambda: defaultdict(int))
    for cls in classes:
        train, val, blind = split_groups(by_class[cls], seed)
        for split, keys in (("train", train), ("val", val), ("blind", blind)):
            for key in keys:
                for src in by_class[cls][key]:
                    # Prefixes make collisions impossible while keeping source
                    # and group information visible during error analysis.
                    dst = out / split / cls / f"{key}_{src.name}"
                    shutil.copy2(src, dst)
                    counts[split][cls] += 1
    return counts


def collect_tiers(root: Path):
    records = []
    zijian = root / "mianliao_zijian"
    mapping = TIER_BY_RANGE
    for folder, cls in mapping.items():
        records.extend((cls, p) for p in images(zijian / folder))
    # The public set has explicit cotton percentages in directory names.
    original = root / "CottonFabricImageBD" / "Original"
    for folder in original.iterdir() if original.exists() else []:
        if not folder.is_dir():
            continue
        m = re.search(r"\d+", folder.name)
        if not m:
            continue
        value = int(m.group())
        cls = "Tier0_Low" if value < 50 else "Tier1_Mid" if value < 80 else "Tier2_High"
        records.extend((cls, p) for p in images(folder))
    return records


def copy_cotton_dataset(root: Path, out: Path, seed: int, real_repeat: int = 8):
    """Build a domain-matched cotton split.

    The fixed 16 self-built fabrics are the historical blind set used by the
    project. Public images are training-only; self-built images are repeated
    in training so the model does not optimize for the easier public domain.
    """
    fixed_blind = {"d2", "d7", "d8", "d10", "d12", "d17",
                   "a1", "a6", "a7", "b3", "b5", "b7", "c1", "c4", "c5", "c7"}
    clear_make(out, TIERS)
    src = root / "mianliao_zijian"
    real = {cls: defaultdict(list) for cls in TIERS}
    for folder, cls in TIER_BY_RANGE.items():
        for image in images(src / folder):
            prefix = image.stem.split("_")[0]
            real[cls][f"{folder}_{prefix}"].append(image)

    rng = random.Random(seed)
    counts = defaultdict(lambda: defaultdict(int))
    for cls in TIERS:
        blind_keys = [key for key in real[cls] if key.rsplit("_", 1)[-1] in fixed_blind]
        pool = [key for key in real[cls] if key not in blind_keys]
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * .20)) if pool else 0
        val_keys, train_keys = pool[:n_val], pool[n_val:]
        for split, keys in (("val", val_keys), ("blind", blind_keys)):
            for key in keys:
                for image in real[cls][key]:
                    copy_one(image, out / split / cls / f"{key}_{image.name}")
                    counts[split][cls] += 1
        for key in train_keys:
            for image in real[cls][key]:
                for repeat in range(real_repeat):
                    suffix = f"__real{repeat}" if repeat else ""
                    copy_one(image, out / "train" / cls / f"{key}_{image.stem}{suffix}{image.suffix}")
                    counts["train"][cls] += 1

    # Every public image is independent training data. Its original folder
    # carries the cotton percentage and is therefore already class-labelled.
    public = root / "CottonFabricImageBD" / "Original"
    for folder in public.iterdir() if public.exists() else []:
        if not folder.is_dir():
            continue
        match = re.search(r"\d+", folder.name)
        if not match:
            continue
        value = int(match.group())
        cls = "Tier0_Low" if value < 50 else "Tier1_Mid" if value < 80 else "Tier2_High"
        for image in images(folder):
            copy_one(image, out / "train" / cls / f"public_{folder.name}_{image.name}")
            counts["train"][cls] += 1
    return counts


def collect_fibers(root: Path):
    records = []
    source = root / "mianliao_5"
    aliases = {"棉": "纯棉", "羊毛": "纯羊毛", "涤纶": "纯涤纶", "丝": "其他", "亚麻": "其他"}
    for folder in source.iterdir() if source.exists() else []:
        cls = aliases.get(folder.name)
        if cls:
            records.extend((cls, p) for p in images(folder))
    # High-cotton, composition-labelled microscope samples are useful pure
    # cotton positives. Keep the mapping explicit so this assumption is easy
    # to replace when laboratory composition labels become available.
    for p in images(root / "mianliao_zijian" / "80-100"):
        records.append(("纯棉", p))
    # Extra non-target material improves rejection, but is never put in the
    # target classes. It is optional and can be omitted on a smaller dataset.
    other = root / "data_other" / "train"
    for p in images(other):
        records.append(("其他", p))
    # The same microscope frame occurs in both m5 and self-built folders under
    # different filenames. Keep only one byte-identical copy so it cannot
    # cross a split boundary or overweight one class.
    seen = set()
    unique = []
    for cls, path in records:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((cls, path))
    return unique


def collect_raw_fibers(root: Path):
    """Collect the original six-class labels used by the existing fiber5 model."""
    records = []
    source = root / "mianliao_5"
    for folder in source.iterdir() if source.exists() else []:
        if folder.name in RAW_FIBERS:
            records.extend((folder.name, p) for p in images(folder))
    # Only the training copy is used. data_other/train and data_other/test are
    # byte-identical; importing both would duplicate the same image.
    records.extend(("其他", p) for p in images(root / "data_other" / "train"))
    return records


def report(name, counts):
    print(f"[{name}]")
    for split in ("train", "val", "blind"):
        print(f"  {split}: " + ", ".join(f"{c}={counts[split][c]}" for c in counts[split]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1].parent),
                    help="数据根目录，需包含 CottonFabricImageBD / mianliao_5 / mianliao_zijian")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1].parent / "dual_data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    tier_counts = copy_cotton_dataset(root, out / "cotton_tier", args.seed)
    fiber_counts = copy_grouped(collect_fibers(root), out / "pure_fiber", FIBERS, args.seed)
    raw_counts = copy_grouped(collect_raw_fibers(root), out / "pure_fiber6", RAW_FIBERS, args.seed)
    report("cotton_tier", tier_counts)
    report("pure_fiber", fiber_counts)
    report("pure_fiber6", raw_counts)
    print(f"数据集已生成: {out}")


if __name__ == "__main__":
    main()
