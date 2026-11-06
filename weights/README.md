# 模型权重说明

## 1. 最终权重

| 用途 | 文件 | 大小 | 输入 | SHA256 |
|---|---|---:|---:|---|
| 棉含量三分类 | `weights/nano_rot90_best_test.pt` | 59,874,891 B | 384 | `67407863594f2edf833b2e148b3aff16c21a672ac1c4e5d2b66be63209817587` |
| 纯面料六分类 | `weights/fiber5_yolov8n_cls_288.pt` | 2,978,760 B | 288 | `e799e230b75ef8c45a9d5ee838bb63439b800d78ec2aacdda8105fee264151d0` |

本地路径：

```text
D:\hunfang\release_final\weights\nano_rot90_best_test.pt
D:\hunfang\release_final\weights\fiber5_yolov8n_cls_288.pt
```

## 2. 部署用模型

树莓派不直接加载 `.pt`，而是加载 NCNN：

```text
packages/64bit/models/cotton_tier/model.ncnn.param|bin
packages/64bit/models/pure_fiber/model.ncnn.param|bin
packages/32bit/models/...
```

NCNN 模型已经随仓库提交，哈希在 `models/manifest.json`。

## 3. 为什么普通 Git 里看不到 `.pt`

`.pt` 是 PyTorch 权重，棉含量模型约 60 MB。普通 Git 每次修改都会保存完整副本，
仓库会快速膨胀，因此使用 Git LFS：

```text
.gitattributes
weights/*.pt filter=lfs diff=lfs merge=lfs -text
```

克隆时需要安装 Git LFS：

```bash
git lfs install
git clone git@github.com:madspotdog/fabric_inspection.git
```

如果只拿到 LFS 指针文件（几十字节的文本），说明没有拉取 LFS 内容：

```bash
git lfs pull
```

## 4. 训练配置

| 模型 | 配置 |
|---|---|
| 棉含量 Nano | `configs/train_cotton_nano.yaml` |
| 纯面料 YOLO | `configs/train_fiber_yolo.yaml` |
| 64 位推理阈值 | `configs/inspection_64bit.json` |
| 32 位推理阈值 | `configs/inspection_32bit.json` |

## 5. 校验

```bash
python tools/verify_models.py
```

校验 NCNN 的 SHA256、`in0/out0` 和类别数。

`.pt` 可用系统工具校验：

```bash
sha256sum weights/nano_rot90_best_test.pt
sha256sum weights/fiber5_yolov8n_cls_288.pt
```

## 6. 许可提醒

- `fiber5_yolov8n_cls_288.pt` 基于 Ultralytics YOLOv8，受 AGPL-3.0 约束。
- `nano_rot90_best_test.pt` 基于 timm ConvNeXt-Nano 微调。
- 商用分发前请阅读 `THIRD_PARTY.md`。
