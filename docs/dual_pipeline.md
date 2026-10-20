# 双模型面料识别

## 任务定义

- `cotton_tier`：`低棉 / 中棉 / 高棉`。
- `pure_fiber`：`纯棉 / 纯羊毛 / 纯涤纶 / 其他`。底层推荐保留 `丝 / 亚麻 / 其他 / 棉 / 涤纶 / 羊毛` 六类，推理时映射到业务四类，拒识更稳。
- 两个独立分类模型并行/顺序执行：棉含量默认使用 `ConvNeXt-Nano`（384 输入，精度优先），纯面料使用已有 `YOLOv8n-cls`（288 输入，速度优先）。低置信度转为 `其他`。
- 交叉规则默认：`低棉 + 其他`、`高棉 + 纯棉` 为“通过”，其他组合为“复核”。规则和阈值均可调。

## 训练

```powershell
python build_dual_datasets.py --root D:\hunfang --out D:\hunfang\dual_data
python train_dual_yolo.py --data D:\hunfang\dual_data --project D:\hunfang\dual_runs --cotton-imgsz 224 --epochs 100 --skip-fiber
python validate_dual.py --data D:\hunfang\dual_data --cotton-model D:\hunfang\student_kd\mjc_convnext_nano_i384\best_test.pt --fiber-model D:\hunfang\deliver\fiber5_yolov8n_cls_288.pt --cotton-imgsz 384 --fiber-imgsz 288
```

训练增强包含 0/90/180/270 度抗旋转、水平/垂直翻转、缩放、颜色扰动和擦除。增强只在训练期在线执行；部署不启用 TTA，不增加现场推理次数。盲测按布料编号/样本组隔离，不能用训练图的增强副本代替盲测。

旋转增强已写入 `train_kd_3tier.py` 和 `train_dual_yolo.py`。当前已导出的树莓派模型仍需在重新训练、固定盲测比较通过后再替换。

`纯棉` 标签当前由 `mianliao_5/棉` 和 `mianliao_zijian/80-100` 组成；`丝`、`亚麻` 进入 `其他`。六类底层模型则保留 `mianliao_5` 的原始类别。若数据库有实验室成分标签，应优先替换这个映射。

## 树莓派 NCNN

在 PC 上分别导出两个 `best.pt`：

```powershell
python export_dual_ncnn.py --cotton-model D:\hunfang\student_kd\mjc_convnext_nano_i384\best_test.pt --fiber-model D:\hunfang\deliver\fiber5_yolov8n_cls_288.pt --out D:\hunfang\rpi_models --cotton-imgsz 384 --fiber-imgsz 288
```

将生成的 `rpi_models/cotton_tier` 和 `rpi_models/pure_fiber` 分别复制到 Pi 项目下的 `models/cotton_tier`、`models/pure_fiber`。输入输出节点保持 `in0/out0`。Pi 只需 `ncnn numpy opencv-python`：

```bash
python3 rpi_dual_inspect.py --check
python3 rpi_dual_inspect.py --image sample.jpg
python3 rpi_dual_inspect.py --camera 0 --threads 2 --tta --cotton-threshold 0.46 --fiber-threshold 0.50
```

`--tta` 启用四向旋转平均，速度约为 4 倍成本，适合盲测或难样本；生产默认关闭并建议使用 3 到 5 张轻微位移/旋转的连拍平均。两个 NCNN 模型分别保留独立概率，便于后续在验证集上调阈值。

阶段性结果见 `dual_validation_report.md`。盲测目标必须用未见过的真实布料和固定阈值复测后确认。
