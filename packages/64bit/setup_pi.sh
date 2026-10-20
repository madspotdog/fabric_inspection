#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libgomp1 fonts-wqy-microhei
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ncnn numpy Pillow
echo "安装完成。先运行: .venv/bin/python rpi_dual_inspect.py --check --threads 2"
