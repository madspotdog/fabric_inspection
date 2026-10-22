# -*- coding: utf-8 -*-
"""校验 NCNN 模型、标签和清单哈希是否一致。

不依赖 ncnn 运行时，只解析 .param 文本，适合 CI 或交付前检查。
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_param(path: Path) -> dict:
    """提取输入/输出节点和最后一层分类数。"""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    has_input = any(line.split()[0] == "Input" and " in0" in line for line in lines)
    has_output = any(" out0" in line for line in lines)
    classes = None
    for line in lines:
        fields = line.split()
        if len(fields) > 5 and fields[0] == "InnerProduct":
            for token in fields[5:]:
                if token.startswith("0="):
                    try:
                        classes = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
    return {"input_in0": has_input, "output_out0": has_output, "classes": classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "models" / "manifest.json")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for name, item in manifest["models"].items():
        param = ROOT / item["param"]
        binary = ROOT / item["bin"]
        print(f"[{name}]")
        for path, key in ((param, "param_sha256"), (binary, "bin_sha256")):
            if not path.is_file():
                failures.append(f"缺少文件: {path}")
                continue
            actual = sha256(path)
            ok = actual == item[key]
            print(f"  {path.name}: {'OK' if ok else 'HASH MISMATCH'}")
            if not ok:
                failures.append(f"{path} 哈希不一致")
        if param.is_file():
            info = parse_param(param)
            print(f"  in0={info['input_in0']} out0={info['output_out0']} classes={info['classes']}")
            if not info["input_in0"] or not info["output_out0"]:
                failures.append(f"{param} 缺少 in0/out0")
            if info["classes"] != len(item["labels"]):
                failures.append(f"{param} 输出类别数与 labels 不一致")
            labels = json.loads((param.parent / "labels.json").read_text(encoding="utf-8"))
            if labels != item["labels"]:
                failures.append(f"{param.parent}/labels.json 与清单不一致")
        if binary.is_file() and binary.stat().st_size != item["bin_bytes"]:
            failures.append(f"{binary} 大小不一致")

    if failures:
        print("\n校验失败:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\n全部校验通过")


if __name__ == "__main__":
    main()
