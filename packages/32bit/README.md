# 树莓派 4B 32 位系统双模型面料识别

适用场景：32 位 Raspberry Pi OS、USB 显微镜、纯 CPU、默认单帧识别。

## 模型

- 棉含量：ConvNeXt-Nano，输入 384，低棉/中棉/高棉。
- 纯面料：YOLOv8n-cls，输入 288，底层六类：丝/亚麻/其他/棉/涤纶/羊毛。
- 业务映射：棉=纯棉，羊毛=纯羊毛，涤纶=纯涤纶，丝/亚麻/其他=其他。
- 交叉规则：低棉+其他通过，高棉+纯棉通过，其余复核。
- 级联门控：纯面料模型 `P(棉)` 达到 `0.20` 才运行棉含量模型；不要求纯面料模型必须直接预测为“棉”。

32 位版本默认关闭 TTA、取消连拍表决、关闭 FP16，优先稳定性和兼容性。每次按空格只执行一帧、两个模型各一次推理。

棉含量模型使用固定中心 `660x660` 显微 ROI。默认最高概率至少 `0.55` 且前两类概率差至少 `0.10` 才直接显示低/中/高棉，否则显示“待定”并进入复核。该设置在固定盲测上覆盖约 88.75% 样本，直接出结果部分准确率约 74.65%。

## 安装

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libgomp1 v4l-utils fonts-wqy-microhei
cd rpi_dual_package_32bit
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果 32 位系统没有可用的 `ncnn` wheel，执行：

```bash
sudo apt install -y build-essential cmake git libopencv-dev
git clone --depth=1 https://github.com/Tencent/ncnn /tmp/ncnn
cd /tmp/ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_PYTHON=ON ..
make -j2
cd ../python
../../rpi_dual_package_32bit/.venv/bin/python -m pip install .
```

## 使用

```bash
.venv/bin/python rpi_dual_inspect.py --check --threads 2 --no-fp16
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --no-fp16 --mjpg --cotton-gate 0.20
```

按空格识别，`q` 或 `ESC` 退出。默认请求 USB 摄像头 `1280x720 MJPG`。

单张图片：

```bash
.venv/bin/python rpi_dual_inspect.py --image sample.jpg --threads 2 --no-fp16
```

不提供默认连拍表决：两方向推理虽然从 67.5% 提升到 72.5%，但耗时约翻倍；四方向只有 71.25%，性价比不适合 32 位树莓派。

## 32 位性能测试

```bash
.venv/bin/python benchmark_dual.py --runs 10 --threads 2 --no-fp16
.venv/bin/python benchmark_dual.py --runs 10 --threads 3 --no-fp16
```

32 位 ARM 的实际 NCNN 速度、温度和内存必须在目标树莓派上测试。建议主动散热、稳定 5V/3A 电源，并用 `vcgencmd get_throttled` 检查降频。
