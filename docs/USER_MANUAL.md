# 用户手册

软件名称：面料显微图像双模型智能识别软件  
版本：V1.0.0

## 1. 软件简介

本软件连接 USB 显微镜，对当面料图像进行两种独立识别：

1. 纯面料类型：纯棉、纯羊毛、纯涤纶、其他
2. 棉含量档位：低棉、中棉、高棉

两个结果经过交叉校验，输出“通过”或“复核”。

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

64 位：

```bash
./run_camera.sh 0
```

32 位：

```bash
./run_camera.sh 0
```

或手动：

```bash
.venv/bin/python rpi_dual_inspect.py \
  --camera 0 --threads 2 --no-fp16 --mjpg \
  --cotton-gate 0.20
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

## 5. 界面说明

摄像头窗口左侧为显微镜画面，绿色方框为中心 660×660 ROI。

右侧面板显示：

```text
面料智能检测
棉含量档位        高棉
置信度            91.0%
纯面料判断        纯棉
置信度            96.0%
棉门控 P(棉)      82.0%
已进入棉含量识别
最终判定          [ 通过 ]
本次推理时间      1200 ms
清晰度            320
```

- 绿色“通过”：两个模型结果符合规则。
- 橙色“复核”：结果不一致、置信度不足或图像质量不足。
- “不适用”：P(棉) 低于门限，未运行棉含量模型。
- “待定”：棉含量模型置信度或类别边界不足。

## 6. 操作

| 按键 | 功能 |
|---|---|
| 空格 | 采集当前帧并识别 |
| Q | 退出 |
| ESC | 退出 |

## 7. 结果解释

### 7.1 通过

```text
低棉 + 其他
高棉 + 纯棉
```

表示两个模型相互支持。

### 7.2 复核

包括：

- 低棉 + 纯棉
- 中棉 + 任意
- 高棉 + 纯涤纶/纯羊毛/其他
- 置信度低于阈值
- 清晰度低于 75

复核不是“判错”，而是提示需要人工确认。

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
| 结果频繁复核 | 检查倍率/对焦；必要时用现场样品重新标定阈值 |

## 12. 维护

- 更换模型后运行 `tools/verify_models.py` 校验哈希和类别数。
- 更换相机/镜头后重新确认 ROI 和清晰度阈值。
- 每月备份台账和样图。
- 不要在同一台账目录运行多个实例。

## 13. 截图清单（申报时补充）

建议在软著说明书中放以下截图：

1. 主界面空闲状态
2. 识别“通过”状态
3. 识别“复核”状态
4. 终端模型自检输出
5. 台账 CSV 示例
6. 安装与启动命令
