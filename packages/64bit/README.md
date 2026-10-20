# 树莓派双模型面料识别

## 模型

- `models/cotton_tier`：ConvNeXt-Nano，棉含量三档，输入 384，ImageNet 归一化。保持 384 是为了保留已验证的 71.25% 盲测结果。
- `models/pure_fiber`：YOLOv8n-cls，六类底层输出：丝、亚麻、其他、棉、涤纶、羊毛，输入 288，0 到 1 归一化。
- 级联门控：只有纯面料模型的 `P(棉)` 达到 `--cotton-gate` 才运行棉含量模型；默认门限为 `0.20`。不要求纯面料模型必须直接预测成“棉”，避免漏掉低棉/混纺。
- 业务映射：棉 -> 纯棉，羊毛 -> 纯羊毛，涤纶 -> 纯涤纶，丝/亚麻/其他 -> 其他。

## 安装

最简单的方式：

```bash
chmod +x setup_pi.sh run_camera.sh
./setup_pi.sh
```

手动安装方式：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libgomp1 fonts-wqy-microhei
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install ncnn numpy Pillow
```

## 使用

```bash
.venv/bin/python rpi_dual_inspect.py --check --threads 2
.venv/bin/python rpi_dual_inspect.py --image sample.jpg --threads 2
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --mjpg \
  --cotton-gate 0.20 --cotton-threshold 0.46 --fiber-threshold 0.50

# 等价的默认启动方式
./run_camera.sh 0
```

空格触发拍照识别，`q` 或 `ESC` 退出。生产默认关闭 TTA，以单次 Nano + 单次 YOLO 推理降低延迟。`--tta` 仍可用于复核，但会让 Nano 棉含量推理约增加 4 倍。

摄像头窗口右侧会显示大字号的棉含量档位、纯面料结果、两路置信度、交叉验证结论和推理耗时；终端仍保留完整概率明细。

USB 显微镜建议先运行下面的基准，分别测试 2/3/4 线程和 FP16：

```bash
.venv/bin/python benchmark_dual.py --runs 10 --threads 2
.venv/bin/python benchmark_dual.py --runs 10 --threads 3
```

## 交叉判定

- `低棉 + 其他`：通过。
- `高棉 + 纯棉`：通过。
- 其他组合：复核。
- 最高置信度低于阈值：输出其他。

阈值可通过 `--cotton-threshold` 和 `--fiber-threshold` 调整。
棉含量门控可通过 `--cotton-gate` 调整。结果中会显示棉概率和是否进入棉含量识别。

Pi 4B 64 位部署建议：主动散热、稳定 5V/3A 电源、USB 相机请求 `1280x720 MJPG`。实际格式和帧率以程序打印的相机协商结果为准。

## 指示灯（三颗 LED）

生产接线（BCM 编号）：

- LED：`GPIO6`、`GPIO17`、`GPIO22`，依次对应第 1/2/3 颗灯（PWM 缓灭，GPIO6 属 0~8 上拉组，略暗属正常）；
- 拍照按钮：`GPIO24`（内部上拉）。

灯的显示逻辑：

- 待机/空闲：三颗灯单点走马灯（亮一颗 → 保持 → 缓慢熄灭 → 下一颗）；
- 检测结果：只亮一颗 —— 第 1/2/3 颗分别对应第 1/2/3 档；
- 异常：三颗同时快闪 3 次。

运行参数：

```bash
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --width 1280 --height 720 --mjpg \
  --fullscreen --leds --led-pins 6,17,22 --button 24 \
  --cotton-gate 0.20 --cotton-threshold 0.46 --fiber-threshold 0.50
```

单独验证灯：

```bash
.venv/bin/python led_test.py --pins 6,17,22
```

## 开机自启

图形桌面（Labwc/Wayland）下用 XDG autostart：

- `/home/<user>/.config/autostart/rpi_dual_v3.desktop` 指向 `autostart.sh`；
- `autostart.sh` 负责 `sleep` 等相机/USB 就绪、启动程序、失败重试，并写 `autostart.log`；
- 走马灯在程序启动后立即开始，相机未就绪会自动重试，不退出。

`autostart.sh` 示例：

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
sleep 12
LOG="$(pwd)/autostart.log"
while true; do
  ./.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --width 1280 --height 720 --mjpg \
    --fullscreen --leds --led-pins 6,17,22 --button 24 \
    --cotton-gate 0.20 --cotton-threshold 0.46 --fiber-threshold 0.50 >>"$LOG" 2>&1
  echo "[$(date)] 程序退出，5 秒后重启" >>"$LOG"
  sleep 5
done
```

界面改造后，右侧面板只显示「判定档位 第 N 档」，与指示灯保持一致；完整概率仍打印到终端日志，便于后续标定。

