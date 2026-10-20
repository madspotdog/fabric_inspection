# -*- coding: utf-8 -*-
"""Raspberry Pi dual-model inference using two NCNN classification models.

The two models are independent: cotton tier (低棉/中棉/高棉) and pure-fiber
candidate (纯棉/纯羊毛/纯涤纶/其他).  A low-confidence result is rejected to
其他.  The two outputs are retained in the JSON-like console line so that a
later calibration pass can change thresholds without retraining.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

COTTON_DEFAULT = ("Tier0_Low", "Tier1_Mid", "Tier2_High")
FIBER_DEFAULT = ("丝", "亚麻", "其他", "棉", "涤纶", "羊毛")

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def cotton_business(name):
    return {"Tier0_Low": "低棉", "低棉": "低棉", "Tier1_Mid": "中棉",
            "混纺": "中棉", "中棉": "中棉", "Tier2_High": "高棉", "高棉": "高棉"}.get(name, "其他")


def fiber_business(name):
    return {"棉": "纯棉", "纯棉": "纯棉", "羊毛": "纯羊毛", "纯羊毛": "纯羊毛",
            "涤纶": "纯涤纶", "纯涤纶": "纯涤纶", "丝": "其他", "亚麻": "其他",
            "其他": "其他"}.get(name, "其他")


def read_labels(model_param, default):
    path = Path(model_param).with_name("labels.json")
    try:
        return tuple(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return tuple(default)


def crop_square(image, size):
    import cv2
    h, w = image.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    crop = image[y:y + side, x:x + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def softmax(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if np.all(values >= -1e-5) and abs(float(values.sum()) - 1.0) < 1e-3:
        return values / max(float(values.sum()), 1e-12)
    values = values - np.max(values)
    p = np.exp(values)
    return p / max(float(p.sum()), 1e-12)


class NcnnClassifier:
    def __init__(self, param, binary, names, size, threads, imagenet_norm=False):
        import ncnn
        self.ncnn, self.names, self.size = ncnn, names, size
        self.imagenet_norm = imagenet_norm
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_packing_layout = True
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(binary)) != 0:
            raise RuntimeError(f"无法加载模型: {param}")

    def probs(self, bgr):
        crop = crop_square(bgr, self.size)
        rgb = crop[:, :, ::-1].copy()
        mat = self.ncnn.Mat.from_pixels(rgb.tobytes(), self.ncnn.Mat.PixelType.PIXEL_RGB,
                                         self.size, self.size)
        if self.imagenet_norm:
            mat.substract_mean_normalize(
                [0.485 * 255, 0.456 * 255, 0.406 * 255],
                [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)],
            )
        else:
            mat.substract_mean_normalize([0, 0, 0], [1 / 255, 1 / 255, 1 / 255])
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        code, out = ex.extract("out0")
        if code != 0:
            raise RuntimeError(f"NCNN 推理失败: {code}")
        return softmax(np.asarray(out))

    def predict(self, bgr, tta=False):
        frames = [bgr]
        if tta:
            frames = [np.rot90(bgr, k).copy() for k in range(4)]
        probs = np.mean([self.probs(frame) for frame in frames], axis=0)
        idx = int(np.argmax(probs))
        return self.names[idx], float(probs[idx]), probs


def inspect(cotton, fiber, image, args):
    t0 = time.perf_counter()
    # Keep each model's independent evidence. Sequential execution avoids two
    # NCNN thread pools fighting for the four Pi cores; this is still one
    # coordinated two-model decision per frame.
    ft_raw, fc, fp = fiber.predict(image, args.tta)
    ft = fiber_business(ft_raw)
    cotton_index = next((i for i, name in enumerate(fiber.names) if name == "棉"), None)
    cotton_gate_prob = float(fp[cotton_index]) if cotton_index is not None else 0.0
    cotton_entered = cotton_gate_prob >= args.cotton_gate
    ct_raw, cc, cp = ("未运行", 0.0, np.zeros(3, dtype=np.float32))
    ct = "不适用"
    if cotton_entered:
        ct_raw, cc, cp = cotton.predict(image, args.tta)
        ct = cotton_business(ct_raw)
        if cc < args.cotton_threshold:
            ct = "待定"
    if fc < args.fiber_threshold:
        ft = "其他"
    accepted = cotton_entered and ((ct == "低棉" and ft == "其他") or (ct == "高棉" and ft == "纯棉"))
    return {
        "cotton_tier": ct, "cotton_raw": ct_raw, "cotton_conf": round(cc, 5),
        "cotton_probs": [round(float(x), 5) for x in cp],
        "cotton_gate_prob": round(cotton_gate_prob, 5),
        "cotton_entered": cotton_entered,
        "pure_fiber": ft, "pure_fiber_raw": ft_raw, "pure_fiber_conf": round(fc, 5),
        "pure_fiber_probs": [round(float(x), 5) for x in fp],
        "cross_check": "通过" if accepted else "复核",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def read_image(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_font(size):
    from PIL import ImageFont
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_panel(image, result):
    """Draw a legible customer-facing result panel without changing inference."""
    import cv2
    from PIL import Image, ImageDraw

    h, w = image.shape[:2]
    panel_w = min(500, max(400, w // 3))
    panel_x = max(0, w - panel_w)
    cv2.rectangle(image, (panel_x, 0), (w, h), (12, 18, 26), -1)
    pil = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    title = load_font(30)
    label = load_font(21)
    value = load_font(36)
    small = load_font(18)
    decision_font = load_font(46)
    x = panel_x + 24
    draw.text((x, 18), "面料智能检测", font=title, fill=(255, 255, 255))
    draw.line((x, 68, w - 24, 68), fill=(105, 125, 145), width=2)

    cotton = result["cotton_tier"]
    fiber = result["pure_fiber"]
    decision = result["cross_check"]
    decision_color = (80, 215, 125) if decision == "通过" else (255, 185, 65)
    rows = (
        ("棉含量档位", cotton, f"置信度 {result['cotton_conf']:.1%}", (90, 190, 255)),
        ("纯面料判断", fiber, f"置信度 {result['pure_fiber_conf']:.1%}", (190, 155, 255)),
    )
    y = 102
    for name, val, conf, color in rows:
        draw.text((x, y), name, font=label, fill=(220, 232, 242))
        draw.text((x, y + 31), val, font=value, fill=(255, 255, 255))
        draw.text((x, y + 78), conf, font=small, fill=(225, 235, 242))
        y += 134
    gate_text = f"棉门控 P(棉) {result['cotton_gate_prob']:.1%}"
    gate_state = "已进入棉含量识别" if result["cotton_entered"] else "未进入棉含量识别"
    draw.text((x, y - 12), gate_text, font=small, fill=(255, 225, 90))
    draw.text((x, y + 14), gate_state, font=small, fill=(225, 235, 242))
    y += 48
    draw.line((x, y - 20, w - 24, y - 20), fill=(105, 125, 145), width=2)
    draw.text((x, y + 2), "最终判定", font=label, fill=(255, 255, 255))
    box_top = y + 34
    box_bottom = box_top + 76
    box_color = (28, 145, 75) if decision == "通过" else (190, 105, 20)
    draw.rounded_rectangle((x, box_top, w - 24, box_bottom), radius=10,
                           fill=box_color, outline=(255, 255, 255), width=2)
    draw.text((x + 18, box_top + 12), decision, font=decision_font, fill=(255, 255, 255))
    time_top = box_bottom + 18
    draw.text((x, time_top), "本次推理时间", font=label, fill=(220, 232, 242))
    draw.text((x, time_top + 27), f"{result['elapsed_ms']:.0f} ms", font=value, fill=(255, 225, 90))
    draw.text((x, h - 42), "SPACE 识别    Q / ESC 退出", font=small, fill=(205, 220, 232))
    return np.asarray(pil)[:, :, ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cotton-param", default=str(here / "models" / "cotton_tier" / "model.ncnn.param"))
    ap.add_argument("--cotton-bin", default=str(here / "models" / "cotton_tier" / "model.ncnn.bin"))
    ap.add_argument("--fiber-param", default=str(here / "models" / "pure_fiber" / "model.ncnn.param"))
    ap.add_argument("--fiber-bin", default=str(here / "models" / "pure_fiber" / "model.ncnn.bin"))
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2,
                    help="每个模型的 NCNN 线程数；两个模型顺序执行，Pi 4B 建议 2")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="同时覆盖两个模型的输入尺寸；不指定时使用 224/288")
    ap.add_argument("--cotton-imgsz", type=int, default=384)
    ap.add_argument("--fiber-imgsz", type=int, default=288)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mjpg", action="store_true", help="请求 USB 相机 MJPG 格式")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cotton-threshold", type=float, default=.46)
    ap.add_argument("--cotton-gate", type=float, default=.20,
                    help="纯面料模型 P(棉) 达到此值才运行棉含量模型")
    ap.add_argument("--fiber-threshold", type=float, default=.50)
    ap.add_argument("--tta", action="store_true", help="四向旋转平均；Nano 棉含量推荐开启")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.imgsz is not None:
        args.cotton_imgsz = args.fiber_imgsz = args.imgsz
    if not (args.image or args.dir or args.camera is not None or args.check):
        raise SystemExit("请指定 --image / --dir / --camera / --check")
    cotton = NcnnClassifier(args.cotton_param, args.cotton_bin,
                            read_labels(args.cotton_param, COTTON_DEFAULT), args.cotton_imgsz, args.threads,
                            imagenet_norm=True)
    fiber = NcnnClassifier(args.fiber_param, args.fiber_bin,
                           read_labels(args.fiber_param, FIBER_DEFAULT), args.fiber_imgsz, args.threads,
                           imagenet_norm=False)
    if args.warmup > 0:
        warm = np.zeros((max(args.cotton_imgsz, args.fiber_imgsz),
                         max(args.cotton_imgsz, args.fiber_imgsz), 3), dtype=np.uint8)
        for _ in range(args.warmup):
            inspect(cotton, fiber, warm, args)
    if args.check:
        image = np.zeros((max(args.cotton_imgsz, args.fiber_imgsz),
                          max(args.cotton_imgsz, args.fiber_imgsz), 3), dtype=np.uint8)
        print(inspect(cotton, fiber, image, args)); return
    if args.image or args.dir:
        files = [args.image] if args.image else sorted(sum(
            [glob.glob(os.path.join(args.dir, ext)) for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
        for path in files:
            image = read_image(path)
            if image is None:
                print(f"{path}: 读取失败"); continue
            print(f"{os.path.basename(path)} -> {inspect(cotton, fiber, image, args)}")
        return
    import cv2
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"空格识别，q/ESC退出；camera={args.camera} requested={args.width}x{args.height} mjpg={args.mjpg}")
    result = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        view = frame.copy()
        cv2.putText(view, "USB microscope", (18, 34), 0, .8, (0, 255, 0), 2)
        if result:
            view = draw_panel(view, result)
        cv2.imshow("dual fabric inspect", view)
        key = cv2.waitKey(1) & 255
        if key in (ord("q"), 27): break
        if key == ord(" "):
            result = inspect(cotton, fiber, frame, args)
            print(result)
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
