# 面料显微图像双模型识别

树莓派 4B（64 位 / 32 位）离线部署包。USB 显微镜采集图像，两个模型独立推理并做交叉校验。

## 流程

```text
USB 显微镜
   ↓
中心 660x660 ROI
   ↓
纯面料模型（YOLOv8n-cls, 288）
   ├─ 输出 P(棉)
   └─ P(棉) >= 0.20 时进入棉含量模型
          ↓
      棉含量模型（ConvNeXt-Nano, 384）
          ↓
      低棉 / 中棉 / 高棉
   ↓
交叉校验：低棉+其他 通过；高棉+纯棉 通过；其余复核
```

- 部署推理默认关闭 TTA，不做连拍表决。
- 训练阶段两个模型都加入了 0/90/180/270 度抗旋转增强。
- 级联门控阈值可用 `--cotton-gate` 调整；设为 `0` 可强制每次都运行棉含量模型。

## 模型

| 用途 | 模型 | 输入 | 权重 |
|---|---|---:|---|
| 棉含量三分类 | ConvNeXt-Nano | 384 | `weights/nano_rot90_best_test.pt` |
| 纯面料六分类 | YOLOv8n-cls | 288 | `weights/fiber5_yolov8n_cls_288.pt` |

树莓派实际使用 `packages/*/models/**/model.ncnn.param|bin`，不依赖 PyTorch。

## 盲测结果

| 模型 | 单图 | 整布表决 |
|---|---:|---:|
| 棉含量（旋转增强 Nano） | 67.50% (54/80) | 75.00% (12/16) |
| 纯面料（Fiber5） | 100% (96/96) | 100% (48/48) |

样本为自建显微镜图，按布料隔离；纯面料结果不能视为独立第三方盲测。

## 文档

- `docs/TRAINING_AND_DEPLOYMENT.md`：完整训练过程、试错记录、超参数、数据预处理与开发板迁移说明。

## 目录

```text
packages/64bit/     64 位树莓派部署包
packages/32bit/     32 位树莓派部署包
scripts/            数据构建、训练、验证、导出脚本
docs/               算法与验收说明
weights/            最终 PyTorch 权重（本地保留，默认不入库）
archives/           两个部署 zip（本地保留，默认不入库）
```

## 树莓派部署

解压对应 `packages/64bit` 或 `packages/32bit` 后：

```bash
chmod +x setup_pi.sh run_camera.sh
./setup_pi.sh
.venv/bin/python rpi_dual_inspect.py --check --threads 2
./run_camera.sh 0
```

32 位系统额外使用 `--no-fp16`；如无 `ncnn` wheel，按包内 README 源码编译。

## 数据

原始数据集（`CottonFabricImageBD`、`mianliao_5`、`mianliao_zijian`、`dual_data` 等）保留在本地工作目录，不提交到仓库。
