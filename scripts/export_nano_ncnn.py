# -*- coding: utf-8 -*-
"""导出 ConvNeXt-Nano(3分类, 384) 到 NCNN param/bin。"""
import os
import shutil
import torch
import timm
import pnnx

ARCH = "convnext_nano.in12k_ft_in1k"
WEIGHT = r"D:\hunfang\student_kd\nano_rot90_i384\best_test.pt"
OUT = r"D:\hunfang\deliver_nano\model"
os.makedirs(OUT, exist_ok=True)

model = timm.create_model(ARCH, pretrained=False, num_classes=3)
state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 384, 384)
pt_path = os.path.join(OUT, "convnext_nano.pt")
pnnx.export(model, pt_path, (dummy,))

# pnnx 产出 convnext_nano.ncnn.param / .bin, 重命名
p = os.path.join(OUT, "convnext_nano.ncnn.param")
b = os.path.join(OUT, "convnext_nano.ncnn.bin")
if os.path.exists(p):
    shutil.move(p, os.path.join(OUT, "convnext_nano.param"))
if os.path.exists(b):
    shutil.move(b, os.path.join(OUT, "convnext_nano.bin"))

param = os.path.join(OUT, "convnext_nano.param")
print("param size:", os.path.getsize(param), "bin size:", os.path.getsize(os.path.join(OUT, "convnext_nano.bin")))
with open(param, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
print("input layer:", lines[2])
print("output layer:", lines[-1])
