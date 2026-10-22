# -*- coding: utf-8 -*-
"""导出版权登记用源程序文本。

常见要求：
- 每页 50 行；
- 不足 60 页时提交全部；
- 超过 60 页时提交前 30 页和后 30 页。

本工具输出：
- source_full.txt       全部源程序
- source_60pages.txt    按 50 行/页切分后的申报文本
- README.txt            行数、页数和文件清单
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    "src/fabric_inspection/*.py",
    "scripts/*.py",
    "packages/64bit/rpi_dual_inspect.py",
    "packages/64bit/benchmark_dual.py",
    "packages/64bit/led_test.py",
    "packages/32bit/rpi_dual_inspect.py",
    "packages/32bit/benchmark_dual.py",
    "tests/*.py",
    "tools/verify_models.py",
)


def collect(root: Path, patterns):
    files = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    unique = []
    seen = set()
    for path in files:
        if path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def clean_lines(path: Path):
    lines = []
    previous_blank = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line:
            if previous_blank:
                continue
            previous_blank = True
        else:
            previous_blank = False
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "copyright")
    ap.add_argument("--lines-per-page", type=int, default=50)
    ap.add_argument("--pages", type=int, default=60)
    args = ap.parse_args()

    files = collect(args.root, DEFAULT_FILES)
    if not files:
        raise SystemExit("没有找到源文件")

    output = []
    for path in files:
        relative = path.relative_to(args.root).as_posix()
        output.append(f"// ===== {relative} =====")
        output.extend(clean_lines(path))
        output.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    full = args.out / "source_full.txt"
    full.write_text("\n".join(output) + "\n", encoding="utf-8")

    per_page = args.lines_per_page
    total_pages = (len(output) + per_page - 1) // per_page
    if total_pages <= args.pages:
        selected = list(range(total_pages))
    else:
        half = args.pages // 2
        selected = list(range(half)) + list(range(total_pages - half, total_pages))

    report = []
    for page in selected:
        start = page * per_page
        chunk = output[start:start + per_page]
        report.append(f"// ---------- 第 {page + 1} 页 ----------")
        report.extend(chunk)
        report.append("")
    (args.out / "source_60pages.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "软著源程序导出说明",
        f"源文件数量: {len(files)}",
        f"有效行数: {len(output)}",
        f"每页行数: {per_page}",
        f"总页数: {total_pages}",
        f"导出页数: {len(selected)}（不足 {args.pages} 页时导出全部）",
        "",
        "source_full.txt 为全部源码；source_60pages.txt 为按申报规则截取的前后页。",
        "提交前请删除本地路径、密钥、个人邮箱等敏感信息。",
        "",
        "文件清单:",
    ]
    readme.extend(f"- {path.relative_to(args.root).as_posix()}" for path in files)
    (args.out / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"文件 {len(files)} 个，有效行 {len(output)}，总页 {total_pages}，导出 {len(selected)} 页")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
