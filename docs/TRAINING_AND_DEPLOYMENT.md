# 训练过程、试错记录与部署参考

本文档记录整个项目的完整训练与选型过程，供后续在树莓派或其他开发板上重新部署、复现或继续优化时参考。

最后更新：2026-10-07

---

## 1. 任务定义

最终系统采用两个独立分类模型 + 业务交叉校验：

```text
USB 显微镜图像
   ↓
中心 660x660 ROI
   ↓
纯面料模型（YOLOv8n-cls, 288）
   │  输出 6 类：丝 / 亚麻 / 其他 / 棉 / 涤纶 / 羊毛
   │  业务映射：棉=纯棉，羊毛=纯羊毛，涤纶=纯涤纶，丝/亚麻/其他=其他
   ↓
读取 P(棉)
   ├─ P(棉) < 0.20：跳过棉含量模型
   └─ P(棉) >= 0.20：运行棉含量模型
            ↓
        ConvNeXt-Nano（384）
        输出：低棉 / 中棉 / 高棉
   ↓
交叉校验
   低棉 + 其他   -> 通过
   高棉 + 纯棉   -> 通过
   其他组合      -> 复核
   置信度不足/失焦 -> 复核
```

说明：

- 棉含量三档来自数据标签约定，不代表模型真的测量了棉含量百分比。
- 纯面料模型输出的“棉”主要来自纯棉样本；低棉/混纺通过 `P(棉)` 门控进入下一阶段，避免被硬分类漏掉。
- 部署阶段不做 TTA、不做连拍表决。抗旋转只发生在训练阶段。

---

## 2. 数据来源

### 2.1 棉含量数据

`CottonFabricImageBD/Original/`，13 个文件夹，约 1300 张，尺寸约 `900x1200`：

```text
30cotton 40cotton 50cotton 53cotton 58cotton 60cotton
63cotton 65cotton 66cotton 80cotton 95cotton 98cotton 99cotton
```

按目录数值映射三档：

| 原始棉含量 | 训练类别 |
|---|---|
| < 50 | Tier0_Low（低棉） |
| 50 ~ 79 | Tier1_Mid（中棉） |
| >= 80 | Tier2_High（高棉） |

`mianliao_zijian/`，46 块自建布料，230 张，尺寸 `2944x1656`，每块布 5 个视角：

| 文件夹 | 布料数 | 图片数 | 类别 |
|---|---:|---:|---|
| `0` | 17（d1-d17） | 85 | Tier0_Low |
| `1-50` | 8（a1-a8） | 40 | Tier0_Low |
| `50-79` | 10（b1-b10） | 50 | Tier1_Mid |
| `80-100` | 11（c1-c11） | 55 | Tier2_High |

关键点：`0/` 是明确不含棉（纯化纤），`1-50/` 是低棉，两者都映射 Tier0。

### 2.2 纯面料数据

`mianliao_5/`，275 张，`2944x1656`：

| 类别 | 数量 |
|---|---:|
| 丝 | 20 |
| 亚麻 | 15 |
| 棉 | 95 |
| 涤纶 | 100 |
| 羊毛 | 45 |

其中 20 张与 `mianliao_zijian/80-100` 字节级重复（d1/d2 与 c3/c4），构建时已按 MD5 去重。

`data_other/`，360 张 `683x512` 的 Crepe 类外部图。`train/` 与 `test/` 是同一批 180 张的重复拷贝，不能同时导入，否则会重复计数并造成泄漏。最终只使用 `train/`。

`data_external/`，294 张 `683x512`：丝 59、亚麻 60、棉 115、涤纶 60。用于补充纯面料类别，但未进入最终盲测。

---

## 3. 数据构建与预处理

最终脚本：`scripts/build_dual_datasets.py`

### 3.1 棉含量数据集

输出 `dual_data/cotton_tier/{train,val,blind}`。

- 固定 16 块自建布料作为历史盲测集：

```text
d2 d7 d8 d10 d12 d17
a1 a6 a7 b3 b5 b7 c1 c4 c5 c7
```

- 其余自建布料按约 20% 划入 val，剩下进入 train。
- 训练集把自建显微图重复 8 份（`__real0..__real7`），提高自建域权重。
- `CottonFabricImageBD/Original` 全部作为 train，不进入 val/blind。
- 盲测和验证集保持原图，不做旋转副本。

最终规模：

| split | Tier0_Low | Tier1_Mid | Tier2_High | 合计 |
|---|---:|---:|---:|---:|
| train | 720 | 940 | 640 | 2300 |
| val | 15 | 5 | 5 | 25 |
| blind | 45 | 15 | 20 | 80 |

### 3.2 纯面料数据集

输出 `dual_data/pure_fiber6/{train,val,blind}`，6 类原始标签：

| split | 丝 | 亚麻 | 其他 | 棉 | 涤纶 | 羊毛 |
|---|---:|---:|---:|---:|---:|---:|
| train | 10 | 5 | 122 | 65 | 70 | 30 |
| val | 5 | 5 | 22 | 10 | 10 | 5 |
| blind | 5 | 5 | 36 | 20 | 20 | 10 |

划分也按布料/样本组隔离，单块布的多个视角不会跨 split。

### 3.3 分组规则（踩过坑）

早期按文件名前缀分组，导致：

- `o_Contrado-Crepe_IMG_xxx-1.jpeg` 全部被当成同一个样本组；
- 训练/验证/盲测分布严重不均；
- 不同目录的字节级重复图跨 split。

最终规则：

- 只有文件名末尾是两位视角编号（`_01`…`_05`）时才去掉编号归为同一组；
- 其他图片以完整文件名作为独立组；
- 构建前做 MD5 去重，重复图只保留一份；
- `data_other/test` 与 `data_other/train` 字节相同，只导入 `train`。

### 3.4 部署预处理

最终树莓派包：

```text
原始 1280x720 画面
   ↓
取中心正方形，边长 min(h, w, 660)
   ↓
棉含量：缩放到 384x384，RGB，ImageNet mean/std
纯面料：缩放到 288x288，RGB，0~1 归一化
```

注意：训练时 Nano 使用 `Resize(384, BICUBIC) + CenterCrop(384)`，部署时是“先裁 660 再缩放”。两者视野比例接近，但插值实现不同（训练 bicubic，部署为 NCNN 双线性/OpenCV area）。如果换相机或倍率，需要重新核对视野一致性。

---

## 4. 试错过程与结果

### 4.1 早期三档 ConvNeXt-Small

权重：`convnext_3tiers_3090_best.pth`（旧资产，已删除）

| 评估 | 单图 | 整布 |
|---|---:|---:|
| val（公开+自建混合） | 81.54% | - |
| test_zijian 单帧 | 66.25% | 75.0%（12/16） |
| test_zijian + 四向 TTA | 60.0% | - |

结论：验证集虚高，自建显微域掉点明显；四向 TTA 在该权重上没有提升。

### 4.2 多次 ConvNeXt-Small 微调

尝试过：

- `cotton_tier_ft_best.pth`
- `cotton_tier_ft_s1/s2.pth`
- `cotton_tier_pub_best.pth`
- `cotton_tier_pubft.pth`
- `cotton_tier_retrain_best.pth`

固定 80 张盲测范围约：

```text
单图 62% ~ 66%
整布 68% ~ 75%
```

结论：在 16 块布、80 张图上继续微调收益有限，且容易对盲测过拟合。

### 4.3 YOLOv8n-cls 三档尝试

| 方案 | val | test_zijian 单图 | 整布 |
|---|---:|---:|---:|
| 224，普通划分 | 85.64% | 30.0% | 25.0% |
| 224，自建过采样 | 80.0% | 45.0% | 50.0% |
| 384，自建过采样 | 86.41% | 40.0% | 37.5% |
| 域匹配重训（自建重复 8 倍） | 60~88% | 32.94% → 45.0% | 43.75% → 50.0% |

结论：YOLOv8n 在公开/自建混合域上很快，但对 16 块自建布料的跨布料泛化不足，不能作为最终棉含量模型。

### 4.4 知识蒸馏轻量模型

教师：`convnext_small.fb_in22k_ft_in1k_384`
学生候选：EfficientNet-B0、MobileNetV3、RepViT、ConvNeXt-Tiny、ConvNeXt-Nano

| 学生 | 单图 | 整布 |
|---|---:|---:|
| EfficientNet-B0（自建+公开） | 61.25% | - |
| ConvNeXt-Tiny | 36.25% | 37.5% |
| ConvNeXt-Nano | 67.5% | 68.75% |
| ConvNeXt-Nano + 四向 TTA | 71.25% | 75.0% |

最终选择 ConvNeXt-Nano：

- 单帧精度与旧 ConvNeXt-Small 持平；
- 参数量约 14.95M，NCNN 权重约 30 MB；
- 比 ConvNeXt-Small 更适合边缘设备；
- 四向 TTA 能到 71.25%，但为了延迟默认不用。

### 4.5 纯面料模型

早期 `fiber3`（棉/涤纶/羊毛）训练出现过类别召回异常；`fiber4`/`fiber4b` 四分类也做过对照。

最终采用：

```text
YOLOv8n-cls，6 类：丝/亚麻/其他/棉/涤纶/羊毛
输入 288，训练 80 轮
```

在 `mianliao_5` 五类原图上映射后：

```text
275/275 = 100%（同源数据，非独立盲测）
```

在 `dual_data/pure_fiber6` 业务盲测上：

```text
96/96 = 100%（其他 46、纯棉 20、纯涤纶 20、纯羊毛 10）
```

注意：这批数据与训练同源，`100%` 不能当作客户现场准确率承诺。

### 4.6 抗旋转重训（最后一次）

要求：两个模型都做 0/90/180/270 训练增强，部署不做 TTA。

实现：

- `scripts/rotation_aug.py`：可序列化的精确正交旋转；
- Nano：`train_kd_3tier.py` 在训练 transform 中加入旋转；
- YOLO：`train_dual_yolo.py` 使用自定义 `ClassificationTrainer`，只对 train split 插入旋转。

结果：

| 模型 | 单图 | 整布 | 是否替换 |
|---|---:|---:|---|
| Nano 旋转增强 | 67.50%（54/80） | 75.00%（12/16） | 是 |
| 旧 Nano | 67.50% | 68.75% | 否 |
| YOLO 旋转增强 | 93.75%（90/96） | 95.83%（46/48） | 否 |
| 旧 Fiber5 | 100%（96/96） | 100% | 保留 |

最终：只替换棉含量 Nano，纯面料继续使用旧 Fiber5。

---

## 5. 最终超参数

### 5.1 ConvNeXt-Nano（棉含量）

```text
架构        convnext_nano.in12k_ft_in1k
输入        384x384
教师        convnext_small.fb_in22k_ft_in1k_384
优化器      AdamW, weight_decay=0.02
学习率      3e-5（最终微调；脚本默认 3e-4）
调度        CosineAnnealingLR, eta_min=1e-6
batch       16
epochs      1 + 5（从旧 Nano best.pt 初始化继续训练）
温度 T      4.0
alpha       0.3（0.3 CE + 0.7 KD）
标签平滑    0.05
类别权重    按训练集逆频率
损失        CrossEntropy + KLDiv
workers     0（Windows 多进程稳定性）
seed        未显式固定（torch/timm 默认）
```

训练增强：

```text
Resize(imgsz*1.04, BICUBIC)
RandomResizedCrop(384, scale=(0.6,1.0), ratio=(0.9,1.1))
随机正交旋转 0/90/180/270
RandomHorizontalFlip
RandomVerticalFlip
RandomGrayscale(p=0.2)
ColorJitter(brightness=0.15, contrast=0.15)
ToTensor + ImageNet Normalize
```

验证/盲测预处理：

```text
Resize(384, BICUBIC)
CenterCrop(384)
ToTensor + ImageNet Normalize
```

### 5.2 YOLOv8n-cls（纯面料）

```text
权重        deliver/fiber5_yolov8n_cls_288.pt（最终保留）
输入        288x288
epochs      80
batch       16
optimizer   auto（Ultralytics 选择 AdamW）
lr0         0.01
degrees     90（训练期旋转）
fliplr      0.5
flipud      0.5
scale       0.9（旧版训练配置）
auto_augment randaugment
erasing     0.4
seed        42
deterministic true
```

抗旋转重训对照配置：

```text
epochs 50, batch 32, imgsz 288
scale 0.35, translate 0.08
hsv_h 0.015, hsv_s 0.35, hsv_v 0.25
erasing 0.2
```

该版本盲测 93.75%，低于旧版 100%，因此未采用。

### 5.3 部署参数

```text
NCNN 线程     2（64/32 位默认）
TTA           关闭
连拍表决      关闭
棉门控        P(棉) >= 0.20
棉含量阈值    最高概率 >= 0.55（32 位）
棉含量边界    前两类概率差 >= 0.10（32 位）
纤维阈值      最高概率 >= 0.50
清晰度阈值    75（低于则复核）
ROI           中心 660x660
摄像头        1280x720 MJPG
32 位 FP16    默认关闭
```

---

## 6. 最终验收指标

固定自建盲测（16 块布、80 张）：

| 模型 | 单图 | 整布表决 |
|---|---:|---:|
| 棉含量 Nano | 67.50% | 75.00% |
| 纯面料 Fiber5 | 100% | 100% |

分级召回（棉含量）：

```text
低棉 32/45 = 71.1%
中棉 10/15 = 66.7%
高棉 15/20 = 75.0%
```

重要边界：

- 棉含量盲测仅 16 块布，统计噪声大；
- 纯面料 100% 来自同源数据划分，不是第三方盲测；
- 现场准确率取决于相机、倍率、照明、对焦与布料批次；
- 正式验收应使用未参与训练的现场样品。

---

## 7. 部署与迁移到其他开发板

### 7.1 目录

```text
packages/64bit/   树莓派 64 位包
packages/32bit/   树莓派 32 位包
scripts/          数据构建、训练、验证、导出
weights/          最终 PyTorch 权重（本地，默认不入库）
```

### 7.2 模型文件

```text
packages/*/models/cotton_tier/model.ncnn.param|bin
packages/*/models/pure_fiber/model.ncnn.param|bin
```

NCNN 输入输出固定为 `in0` / `out0`。

### 7.3 运行

```bash
chmod +x setup_pi.sh run_camera.sh
./setup_pi.sh

# 自检
.venv/bin/python rpi_dual_inspect.py --check --threads 2

# 64 位
./run_camera.sh 0

# 32 位（默认 no-fp16）
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --no-fp16 --mjpg
```

### 7.4 换开发板注意事项

1. NCNN 需要能安装 Python 绑定；32 位 ARM 可能没有 wheel，需要源码编译。
2. 先用 `benchmark_dual.py` 测单帧耗时，再决定线程数。
3. 树莓派 4B 无 ARM FP16 硬件指令时，`--no-fp16` 往往更快。
4. 其他开发板（RK3588、Jetson、x86）可优先尝试 ONNX Runtime 或 NCNN Vulkan，但需重新验证输出一致性。
5. 若更换相机分辨率或倍率，必须重新核对中心 ROI 与训练视野比例。
6. 若更换镜头导致方向不固定，优先启用训练增强模型，不要依赖推理 TTA。
7. 现场应保存原始 660x660 ROI 图片和真实标签，才能做有效误差归因。

### 7.5 复现训练

```bash
python scripts/build_dual_datasets.py --root <数据根目录> --out dual_data

# 棉含量（需要教师权重和初始化权重）
python scripts/train_kd_3tier.py --student convnext_nano.in12k_ft_in1k \
  --imgsz 384 --epochs 6 --batch 16 --lr 3e-5 \
  --data dual_data/cotton_tier --testdir dual_data/cotton_tier/blind \
  --name nano_rot90 --init <旧Nano权重> --workers 0

# 纯面料
python scripts/train_dual_yolo.py --data dual_data \
  --project dual_runs_rot90 --fiber-imgsz 288 --epochs 50 --batch 32 \
  --skip-cotton --fiber-model <Fiber5权重> \
  --fiber-data dual_data/pure_fiber6 --fiber-name pure_fiber6_rot90

# 固定盲测对比
python scripts/validate_dual.py --data dual_data \
  --cotton-model <Nano权重> --fiber-model <Fiber权重> \
  --cotton-imgsz 384 --fiber-imgsz 288
```

### 7.6 导出 NCNN

```bash
python scripts/export_nano_ncnn.py     # Nano -> NCNN
python scripts/export_dual_ncnn.py     # Fiber5 -> NCNN
```

导出后必须用包内 `--check` 做一次真实前向，确认 `in0/out0` 与类别顺序。

---

## 8. 已知问题与结论

1. 最大风险不是模型大小，而是域差异：公开 900x1200 图与自建 2944x1656 显微图分布在早期造成严重掉点。
2. 验证集准确率高不代表现场准确；必须按布料隔离盲测。
3. 数据泄漏比模型选择影响更大：重复图、同名文件、train/test 同源都要先处理。
4. 棉含量三档在 16 块布上已接近当前数据上限，继续调模型收益有限，优先补数据。
5. 旋转增强对整布表决有帮助，对单图提升有限；纯面料旋转重训反而下降，故未替换。
6. 级联门控能降低平均延迟，但门控误判会让棉含量模型不运行；演示阶段可用 `--cotton-gate 0` 观察两路独立结果。
7. 交叉校验只能标记“不一致/复核”，不能在无真值情况下判断哪个模型正确。
