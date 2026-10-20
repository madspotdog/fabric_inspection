# -*- coding: utf-8 -*-
"""三颗LED指示灯自检：走马灯 → 1/2/3档单灯 → 三闪。

默认引脚为 BCM 6,17,22（与生产参数 --led-pins 6,17,22 一致），可用 --pins 修改。
用法：.venv/bin/python led_test.py            # 默认 BCM 6,17,22
      .venv/bin/python led_test.py --pins 17,27,22
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("--pins", default="6,17,22", help="BCM 引脚，逗号分隔，依次对应第1~3颗灯")
ap.add_argument("--cycles", type=int, default=3, help="走马灯循环次数")
args = ap.parse_args()

try:
    from gpiozero import PWMLED
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"未找到 gpiozero: {exc}\n请先安装：sudo apt install -y python3-gpiozero")

pins = [int(p) for p in args.pins.split(",") if p.strip()]
leds = [PWMLED(p) for p in pins]


def all_off():
    for led in leds:
        led.value = 0.0


def chase():
    seq = list(range(len(leds))) + list(range(len(leds) - 2, 0, -1))
    for idx in seq:
        led = leds[idx]
        led.value = 1.0
        time.sleep(0.30)
        for v in (0.8, 0.6, 0.4, 0.2, 0.0):
            led.value = v
            time.sleep(0.03)
        led.value = 0.0


def level(n):
    all_off()
    if 1 <= n <= len(leds):
        leds[n - 1].value = 1.0


try:
    print("走马灯...")
    for _ in range(args.cycles):
        chase()

    print("依次点亮第 1/2/3 颗...")
    for n in range(1, len(leds) + 1):
        level(n)
        print(f"  第 {n} 档 -> 亮第 {n} 颗")
        time.sleep(0.8)
        all_off()
        time.sleep(0.3)

    print("异常三闪...")
    for _ in range(3):
        for led in leds:
            led.value = 1.0
        time.sleep(0.22)
        all_off()
        time.sleep(0.22)

    print("LED 自检完成")
finally:
    all_off()
