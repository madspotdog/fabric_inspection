import argparse
import time
from pathlib import Path

import numpy as np

from rpi_dual_inspect import Net, labels, COTTON_DEFAULT, FIBER_DEFAULT, inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent / "models"
    fp16 = not args.no_fp16
    cotton_param = root / "cotton_tier" / "model.ncnn.param"
    fiber_param = root / "pure_fiber" / "model.ncnn.param"
    cotton = Net(str(cotton_param), labels(cotton_param, COTTON_DEFAULT),
                 384, args.threads, True, fp16)
    fiber = Net(str(fiber_param), labels(fiber_param, FIBER_DEFAULT),
                288, args.threads, False, fp16)
    image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    config = argparse.Namespace(cotton_threshold=0.0, cotton_margin=0.0,
                                cotton_gate=0.0, fiber_threshold=0.0, focus_threshold=0.0)
    for _ in range(2):
        inspect(cotton, fiber, image, config)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        inspect(cotton, fiber, image, config)
        times.append(time.perf_counter() - start)
    print(f"threads={args.threads} fp16={fp16} mean={np.mean(times):.3f}s "
          f"min={min(times):.3f}s max={max(times):.3f}s fps={1 / np.mean(times):.2f}")


if __name__ == "__main__":
    main()
