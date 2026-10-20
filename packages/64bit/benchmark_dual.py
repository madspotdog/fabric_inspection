# -*- coding: utf-8 -*-
"""Benchmark the two NCNN models on the actual Raspberry Pi."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rpi_dual_inspect import NcnnClassifier, COTTON_DEFAULT, FIBER_DEFAULT, inspect, read_labels


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    cotton_param = here / "models/cotton_tier/model.ncnn.param"
    fiber_param = here / "models/pure_fiber/model.ncnn.param"
    cotton = NcnnClassifier(str(cotton_param), str(cotton_param.with_suffix(".bin")),
                            read_labels(cotton_param, COTTON_DEFAULT), 384, args.threads, True)
    fiber = NcnnClassifier(str(fiber_param), str(fiber_param.with_suffix(".bin")),
                           read_labels(fiber_param, FIBER_DEFAULT), 288, args.threads, False)
    image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    config = argparse.Namespace(tta=args.tta, cotton_threshold=0.0, cotton_gate=0.0, fiber_threshold=0.0)
    for _ in range(2):
        inspect(cotton, fiber, image, config)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        inspect(cotton, fiber, image, config)
        times.append(time.perf_counter() - start)
    print(f"threads={args.threads} tta={args.tta} runs={args.runs}")
    print(f"mean={np.mean(times):.3f}s min={np.min(times):.3f}s max={np.max(times):.3f}s fps={1/np.mean(times):.2f}")


if __name__ == "__main__":
    main()
