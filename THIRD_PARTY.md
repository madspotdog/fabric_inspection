# 第三方组件与许可证说明

本文件用于交付前合规审查。**软件著作权登记不替代第三方许可证审查。**

## 1. 运行时与训练依赖

| 组件 | 用途 | 许可证 | 备注 |
|---|---|---|---|
| Ultralytics YOLOv8 | 纯面料模型训练/导出 | **AGPL-3.0** | 商用分发需购买 Ultralytics 商业许可，或按 AGPL 开源整个衍生作品 |
| NCNN | 边缘推理 | BSD-3-Clause | 需保留版权声明 |
| timm | ConvNeXt 模型与权重 | Apache-2.0 | 预训练权重以其模型卡许可证为准 |
| PyTorch | 训练 | BSD-3-Clause | 需保留版权声明 |
| torchvision | 图像变换 | BSD-3-Clause | 需保留版权声明 |
| OpenCV | 采集/预处理 | Apache-2.0 | 需保留版权声明 |
| NumPy | 数值计算 | BSD-3-Clause | 需保留版权声明 |
| Pillow | 中文字体绘制 | HPND | 需保留版权声明 |
| Python | 运行时 | PSF License | 需保留版权声明 |

## 2. 重点风险：YOLOv8 的 AGPL-3.0

当前纯面料模型由 Ultralytics YOLOv8n-cls 训练和导出。AGPL-3.0 的核心义务：

- 分发软件时需要提供对应源代码；
- 通过网络提供服务时，也可能需要向用户提供源代码；
- 衍生作品需要以 AGPL 兼容方式授权。

可选方案：

1. 购买 Ultralytics 商业许可证；
2. 将纯面料模型替换为 Apache-2.0/BSD 许可的架构（如 timm 中的 MobileNetV3、EfficientNet）并重新训练；
3. 按 AGPL 要求开源相应部分。

**在完成上述选择前，不建议对外销售或闭源分发。**

## 3. 数据集权利

| 数据集 | 来源 | 风险 |
|---|---|---|
| CottonFabricImageBD | 项目历史数据 | 需确认原始授权和商用范围 |
| mianliao_5 / mianliao_zijian | 自建拍摄 | 版权归拍摄方，需保留授权文件 |
| data_other / data_external | 外部公开图片 | 需逐项确认来源和许可 |

数据集未提交到 Git 仓库，但训练出的模型可能包含其分布特征。商用前应确认数据授权。

## 4. 模型权重

- `weights/nano_rot90_best_test.pt`：本项目基于 timm ConvNeXt-Nano 微调。
- `weights/fiber5_yolov8n_cls_288.pt`：基于 Ultralytics YOLOv8n-cls 微调，受 AGPL 约束。
- NCNN 模型由上述权重导出，分发时同样需要考虑上游许可证。

## 5. 交付建议

1. 在软件说明书中声明使用的第三方组件和许可证。
2. 随软件附带本文件和原始许可证文本。
3. 对外分发前完成 AGPL 决策。
4. 保留数据集授权、拍摄记录和模型训练记录。
5. 软著登记材料中不要包含第三方密钥、账号、个人路径。
