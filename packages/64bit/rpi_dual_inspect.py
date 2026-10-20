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
    return {"棉": "棉", "纯棉": "棉", "羊毛": "羊毛", "纯羊毛": "羊毛",
            "涤纶": "涤纶", "纯涤纶": "涤纶", "丝": "其他", "亚麻": "其他",
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
    accepted = cotton_entered and ((ct == "低棉" and ft == "其他") or (ct == "高棉" and ft == "棉"))
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
    """只显示与指示灯一致的"判定档位"。"""
    import cv2
    from PIL import Image, ImageDraw

    h, w = image.shape[:2]
    panel_w = min(520, max(420, w // 3))
    panel_x = max(0, w - panel_w)
    cv2.rectangle(image, (panel_x, 0), (w, h), (12, 18, 26), -1)
    pil = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    title = load_font(30)
    label = load_font(24)
    big = load_font(84)
    small = load_font(19)
    x = panel_x + 24

    level = result_level(result)
    meaning = {1: "其他", 2: "羊毛 / 涤纶 / 低棉", 3: "棉"}.get(level, "其他")

    draw.text((x, 20), "面料智能检测", font=title, fill=(255, 255, 255))
    draw.line((x, 72, w - 24, 72), fill=(105, 125, 145), width=2)
    draw.text((x, 130), "判定档位", font=label, fill=(220, 232, 242))
    draw.text((x, 170), f"第 {level} 档", font=big, fill=(0, 220, 0))
    draw.text((x, 300), meaning, font=label, fill=(190, 205, 215))
    draw.line((x, h - 152, w - 24, h - 152), fill=(105, 125, 145), width=2)
    draw.text((x, h - 130), "本次推理时间", font=label, fill=(220, 232, 242))
    draw.text((x, h - 94), f"{result['elapsed_ms']:.0f} ms", font=load_font(40), fill=(255, 225, 90))
    draw.text((x, h - 42), "SPACE 识别    Q / ESC 退出", font=small, fill=(205, 220, 232))
    return np.asarray(pil)[:, :, ::-1].copy()


class LedBar:
    """三颗 LED 指示灯（BCM 引脚，PWM 支持缓灭）。没有 GPIO 库或在非树莓派上时自动禁用。"""

    def __init__(self, pins, enabled=True):
        self.enabled = bool(enabled and pins)
        self.leds = []
        self._bri = []
        self._stop = None
        self._thread = None
        if self.enabled:
            try:
                from gpiozero import PWMLED
                self.leds = [PWMLED(int(p)) for p in pins]
                self._bri = [0.0] * len(self.leds)
            except Exception as exc:
                print(f"[LED] 初始化失败(已禁用): {exc}")
                self.enabled = False

    def _write(self, i, v):
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        self._bri[i] = v
        self.leds[i].value = v

    def set_level(self, level):
        """结果灯：只亮第 level 颗（1→第1颗, 2→第2颗, 3→第3颗），其余全灭。"""
        if not self.enabled:
            return
        for i in range(len(self.leds)):
            self._write(i, 1.0 if i == (level - 1) else 0.0)

    def all_off(self):
        if not self.enabled:
            return
        for i in range(len(self.leds)):
            self._write(i, 0.0)

    def start_loop(self):
        """空闲单点走马灯：点亮一颗 → 保持 → 缓慢熄灭 → 熄灭后下一颗再亮（不重叠）。"""
        if not self.enabled:
            return
        import threading
        import time
        self._stop = threading.Event()

        def anim():
            n = len(self.leds)
            seq = list(range(n)) + list(range(n - 2, 0, -1))
            dt = 0.02
            up, down, hold = 0.10, 0.04, 0.25   # 渐亮/渐灭步长、全亮保持时长
            while not self._stop.is_set():
                for cur in seq:
                    if self._stop.is_set():
                        return
                    v = 0.0
                    while v < 1.0 and not self._stop.is_set():
                        v = min(1.0, v + up)
                        self._write(cur, v)
                        time.sleep(dt)
                    t0 = time.time()
                    while (time.time() - t0) < hold and not self._stop.is_set():
                        time.sleep(dt)
                    while v > 0.0 and not self._stop.is_set():
                        v = max(0.0, v - down)
                        self._write(cur, v)
                        time.sleep(dt)
                    self._write(cur, 0.0)

        self._thread = threading.Thread(target=anim, daemon=True)
        self._thread.start()

    def stop_loop(self):
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
        if getattr(self, "_thread", None) is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._stop = None

    def error(self):
        if not self.enabled:
            return
        self.stop_loop()
        import time
        for _ in range(3):
            for i in range(len(self.leds)):
                self._write(i, 1.0)
            time.sleep(0.22)
            for i in range(len(self.leds)):
                self._write(i, 0.0)
            time.sleep(0.22)


def result_level(result):
    """按规则把结果映射成 1/2/3 颗灯。"""
    ct = result.get("cotton_tier")
    ft = result.get("pure_fiber")
    if ct in ("中棉", "高棉"):
        return 3
    if ct == "低棉":
        return 2
    if ft == "棉":
        return 3
    if ft in ("羊毛", "涤纶"):
        return 2
    return 1


def blur_score(bgr):
    """拉普拉斯方差，越大越清晰。"""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
    ap.add_argument("--fullscreen", action="store_true", help="全屏显示窗口")
    ap.add_argument("--leds", action="store_true", help="用三颗LED(默认BCM17/27/22)显示结果")
    ap.add_argument("--led-pins", default="17,27,22", help="LED的GPIO(BCM)，逗号分隔，依次对应1~3颗灯")
    ap.add_argument("--blur-thresh", type=float, default=0.0,
                    help="拉普拉斯方差低于此值判为模糊->其他(1灯)；0=关闭")
    ap.add_argument("--button", type=int, default=0, help="拍照按钮的GPIO(BCM)，0=关闭")
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
    pins = [p.strip() for p in args.led_pins.split(",") if p.strip()]
    ledbar = LedBar(pins, args.leds)
    btn = None
    if args.button:
        try:
            from gpiozero import Button
            btn = Button(args.button, pull_up=True, bounce_time=0.05)
        except Exception as exc:
            print(f"[按钮] 初始化失败(已禁用): {exc}")
            btn = None
    print(f"空格/按钮拍照，q/ESC退出；camera={args.camera} {args.width}x{args.height} mjpg={args.mjpg} fullscreen={args.fullscreen} leds={args.leds}")

    ledbar.start_loop()          # 开机立即走马灯（空闲提示），不依赖相机是否就绪
    win = "dual fabric inspect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    result = None
    fs_done = False
    btn_last = False
    cap = None
    try:
        while True:
            if cap is None:
                cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = None
                    if (cv2.waitKey(1) & 255) in (ord("q"), 27):
                        break
                    time.sleep(1.5)
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                if args.mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                print("摄像头已就绪")
            ok, frame = cap.read()
            if not ok:
                cap.release(); cap = None
                continue
            view = frame.copy()
            cv2.putText(view, "USB microscope", (18, 34), 0, .8, (0, 255, 0), 2)
            if result:
                view = draw_panel(view, result)
            cv2.imshow(win, view)
            if args.fullscreen and not fs_done:
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fs_done = True
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27): break
            trigger = (key == ord(" "))
            if btn is not None:
                p = btn.is_pressed
                if p and not btn_last:
                    trigger = True
                btn_last = p
            if trigger:
                result = inspect(cotton, fiber, frame, args)
                print(result)
                level = result_level(result)
                ledbar.stop_loop()
                ledbar.set_level(level)
                print(f"[灯] 亮 {level} 颗")
    except Exception as exc:
        print("[错误]", exc)
        ledbar.error()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        ledbar.stop_loop()
        ledbar.all_off()


if __name__ == "__main__":
    main()
