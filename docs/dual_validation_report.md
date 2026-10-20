# 双模型阶段性验收

日期：2026-10-07

## 最终候选

| 模型 | 权重 | 输入 | 盲测单图 | 整布表决 |
|---|---|---:|---:|---:|
| 棉含量三分类 | `student_kd/mjc_convnext_nano_i384/best_test.pt` | 384 | 57/80 = **71.25%** | 12/16 = **75.00%** |
| 纯面料六分类，业务映射为四类 | `deliver/fiber5_yolov8n_cls_288.pt` | 288 | 96/96 = **100.00%** | 48/48 = **100.00%** |

棉含量 Nano 盲测按固定 16 块自建布料划分，启用四向旋转 TTA；纯面料模型在 `mianliao_5` 及对应业务盲测集上评估。纯面料底层类别为丝、亚麻、其他、棉、涤纶、羊毛，业务层映射为其他、纯棉、纯涤纶、纯羊毛。

## 旋转增强重训结果

- Nano 旋转增强版：单图 `54/80 = 67.50%`，整布 `12/16 = 75.00%`，与旧版单图持平、整布更好，已更新到两个树莓派包。
- YOLOv8n 旋转增强版：`90/96 = 93.75%`，低于旧 Fiber5 模型 `96/96 = 100%`，因此没有替换纯面料模型。
- 部署推理仍不开 TTA；旋转只发生在训练阶段。

## 交叉判定

- `低棉 + 其他`：通过。
- `高棉 + 纯棉`：通过。
- 其他组合：复核。
- 棉含量置信度低于 `--cotton-threshold`，或纯面料置信度低于 `--fiber-threshold`：对应结果转为其他。

## 树莓派交付

已生成目录：`rpi_dual_package/`

- `models/cotton_tier/model.ncnn.param/bin`：Nano 棉含量模型。
- `models/pure_fiber/model.ncnn.param/bin`：YOLOv8n 纯面料模型。
- 两个模型均已检查 `in0/out0` 节点。
- PC 端 NCNN 双路自检通过，预热后示例合成图双路耗时约 105 ms；该数值不是树莓派实测结果。

树莓派执行：

```bash
.venv/bin/python rpi_dual_inspect.py --check --threads 2
.venv/bin/python rpi_dual_inspect.py --camera 0 --threads 2 --mjpg \
  --cotton-threshold 0.46 --fiber-threshold 0.50
```

生产默认关闭 TTA，以降低 Nano 棉含量模型延迟；树莓派最终仍需实测线程数、温度、内存、USB 摄像头格式和实际图像质量。当前盲测已经超过 70% 目标，但样本量只有 16 块自建布料，建议上线前继续增加不同批次布料复核。
