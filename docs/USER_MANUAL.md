# 用户手册

软件名称：纯面料及混纺双模型智能识别软件  
版本：V1.0.0

## 1. 软件简介

本软件连接 USB 显微镜，对面料图像进行两种独立识别：

1. 纯面料类型：棉、羊毛、涤纶、丝、亚麻、其他；
2. 混纺棉含量档位：低棉、中棉、高棉。

最终结果统一映射为**判定档位（第 1 / 2 / 3 档）**，通过**三颗指示灯**和屏幕面板同时输出，便于现场快速读取并据此调节熨烫温度。

> 说明：开发过程中的“通过 / 复核”是中间功能，最终交付版以**档位 + 三灯**作为输出，不再以“通过 / 复核”作为主界面结果。

## 2. 安装

### 2.1 64 位树莓派

```bash
cd packages/64bit
chmod +x setup_pi.sh run_camera.sh
./setup_pi.sh
```

### 2.2 32 位树莓派

```bash
cd packages/32bit
chmod +x setup_pi.sh run_camera.sh
./setup_pi.sh
```

如果提示找不到 `ncnn`，按 32 位包 README 中的源码编译步骤安装。

## 3. 启动前检查

```bash
.venv/bin/python rpi_dual_inspect.py --check --threads 2
```

64 位正常输出一行 JSON；32 位增加 `--no-fp16`。

检查摄像头：

```bash
v4l2-ctl --list-devices
```

确认设备号，通常是 `/dev/video0`。

## 4. 运行

### 4.1 摄像头

```bash
./run_camera.sh 0
```

或手动：

```bash
.venv/bin/python rpi_dual_inspect.py \
  --camera 0 --threads 2 --mjpg --cotton-gate 0.20
```

### 4.2 单张图片

```bash
.venv/bin/python rpi_dual_inspect.py \
  --image sample.jpg --threads 2
```

### 4.3 批量目录

```bash
.venv/bin/python rpi_dual_inspect.py \
  --dir ./images --threads 2
```

### 4.4 指示灯与按钮（生产）

```bash
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --width 1280 --height 720 \
  --mjpg --fullscreen --leds --led-pins 6,17,22 --button 24 \
  --cotton-gate 0.20 --cotton-threshold 0.46 --fiber-threshold 0.50
```

- 待机：三颗灯单点走马灯；
- 结果：只亮一颗（第 1/2/3 颗对应第 1/2/3 档）；
- 异常：三颗同时快闪 3 次；
- 校验灯：`python led_test.py --pins 6,17,22`。

## 5. 界面说明

摄像头窗口左侧为显微镜画面，右侧面板显示：

```text
面料智能检测
判定档位
第 N 档
（含义：其他 / 羊毛·涤纶·低棉 / 棉）
本次推理时间   1200 ms
SPACE 识别    Q / ESC 退出
```

- “判定档位 第 N 档”与三颗指示灯保持一致；
- 面板与指示灯只显示最终档位，完整概率打印在终端日志中。

## 6. 操作

| 按键 | 功能 |
|---|---|
| 空格 / 机身按钮 | 采集当前帧并识别 |
| Q | 退出 |
| ESC | 退出 |

## 7. 结果解释

最终结果映射为三档，指示灯与屏幕一致：

| 档位 | 亮的灯 | 含义 |
|---|---|---|
| 第 1 档 | 第 1 颗 | 其他 |
| 第 2 档 | 第 2 颗 | 羊毛 / 涤纶 / 低棉 |
| 第 3 档 | 第 3 颗 | 棉 |

判定规则（程序内部）：

- 棉含量为“中棉 / 高棉”→ 第 3 档；
- 棉含量为“低棉”→ 第 2 档；
- 未进入棉含量识别、但面料预测为“棉”→ 第 3 档；
- 面料预测为“羊毛 / 涤纶”→ 第 2 档；
- 其他情况 → 第 1 档。

## 8. 参数调整

| 参数 | 作用 | 建议 |
|---|---|---|
| `--cotton-gate` | P(棉) 门限 | 默认 0.20；演示可设 0 强制双模型 |
| `--cotton-threshold` | 棉含量置信度 | 64 位 0.46，32 位 0.55 |
| `--cotton-margin` | 前两类概率差 | 32 位 0.10 |
| `--fiber-threshold` | 纯面料置信度 | 0.50 |
| `--focus-threshold` | 清晰度阈值 | 75 |
| `--threads` | 推理线程 | 2 起步，实测 2/3/4 |
| `--no-fp16` | 关闭 FP16 | 32 位默认开启 |
| `--leds` / `--led-pins` | 启用三灯并指定引脚 | 生产用 6,17,22 |
| `--button` | 拍照按钮引脚 | 生产用 24 |

## 9. 性能测试

```bash
.venv/bin/python benchmark_dual.py --runs 10 --threads 2
```

比较 2/3/4 线程和 FP16 开关，选择平均耗时最低且温度稳定的组合。

## 10. 台账

每次识别可写入：

```text
inspection_records/
├─ inspection_report.csv
└─ Sample001_xxx.jpg
```

CSV 为 UTF-8 BOM，字段见软件设计说明书。

## 11. 常见问题

| 现象 | 处理 |
|---|---|
| 模型加载失败 | 检查 `models/*/model.ncnn.param|bin` |
| 打不开摄像头 | 查 `/dev/video0`，换 `--camera 1` |
| 画面很暗/模糊 | 调焦、调光，确清晰度大于 75 |
| 中文显示方块 | 安装 `fonts-wqy-microhei` |
| 推理很慢 | 降低线程争抢、检查散热、试 `--no-fp16` |
| 指示灯不亮 | 检查 `--leds` 与引脚；运行 `led_test.py` |

## 12. 维护

- 更换模型后运行 `tools/verify_models.py` 校验哈希和类别数。
- 更换相机/镜头后重新确认 ROI 和清晰度阈值。
- 每月备份台账和样图。
- 不要在同一台账目录运行多个实例。

## 13. 截图清单（申报时补充）

建议在软著说明书中放以下截图：

1. 主界面空闲（三灯走马灯）
2. 识别结果：第 1 档
3. 识别结果：第 2 档 / 第 3 档
4. 终端模型自检输出
5. 台账 CSV 示例
6. 安装与启动命令
