面料微观光学质检项目：AI Agent 基准复现与边缘部署交接指南本说明文件供接手本项目的 AI Agent（或研发工程师）使用。文档明确了现有模型的基准指标、评估前处理协议、性能回退归因以及树莓派端侧部署的技术规范。1. 核心技术基线与资产任务定义：面料微观显微图像棉含量三档定性识别（Tier 0: $<50\%$ 低棉/化纤；Tier 1: $50\% \sim 70\%$ 混纺棉；Tier 2: $>80\%$ 高品质纯棉）。模型架构：convnext_small.fb_in22k_ft_in1k_384（三分类，输入分辨率 $384 \times 384$）。权重文件：convnext_3tiers_3090_best.pth。基准性能指标：实验室验证集 (val, 390 样本)：Top-1 准确率 $81.54\%$（Tier 0 召回 $91.7\%$，Tier 1 召回 $76.7\%$，Tier 2 召回 $85.0\%$）。自建真实盲测集 (test_zijian, 80 样本)：标准基准 Top-1 准确率应为 $\ge 75.0\%$。2. 盲测集精度掉至 66.25% 的问题诊断与约束若在 test_zijian（80 样本）上跑出的评测结果仅为 $66.25\%$（特别是 Tier 2 召回率断崖式跌至 $45.0\%$），禁止盲目重新训练权重。这是由于评测脚本未严格遵循既定的显微前处理与推理增强协议导致的。核心诱因剖析缺失 4 向正交旋转 TTA（Test-Time Augmentation）：显微镜下纤维的“天然转曲”和阴影具有强方向性。先前的 $75\%$ 达标指标是基于 $0^\circ, 90^\circ, 180^\circ, 270^\circ$ 四向旋转推理并取 Softmax 平均概率输出。若降级为单帧 $0^\circ$ 推理，模型对丝光棉与高支棉纤维的形态特征捕捉严重不足，直接造成 Tier 2 漏检。输入尺寸失真（Direct Resize vs CenterCrop）：显微相机多为 $1280 \times 720$ 或 $4:3$ 比例。严禁直接拉伸至 $384 \times 384$（会导致纤维微观直径及转曲周期几何形变）。必须先提取中心正方形视场，再进行缩放。插值算法退化：ConvNeXt 训练标准为双三次插值（Bicubic）。若降级为双线性插值（Bilinear），会导致纤维边缘高频毛羽特征过度平滑。3. 标准评估复现脚本 (eval_benchmark.py)接手的 AI Agent 请直接运行以下标准化评估脚本，对齐基准评测管线：import os
from PIL import Image
import timm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ==================== 1. 运行配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = r"convnext_3tiers_3090_best.pth"
TEST_DIR = r"test_zijian"  # 包含 Tier0, Tier1, Tier2 子文件夹

# ==================== 2. 模型与标准化预处理 ====================
model = timm.create_model(
    "convnext_small.fb_in22k_ft_in1k_384", pretrained=False, num_classes=3
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 严格采用 Bicubic 插值与 CenterCrop
base_transform = T.Compose([
    T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(384),
    T.ToTensor>,
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==================== 3. 4 向正交旋转 TTA 推理 ====================
def predict_with_4tta(pil_img):
  tensor_0 = base_transform(pil_img)
  tensor_90 = base_transform(TF.rotate(pil_img, 90))
  tensor_180 = base_transform(TF.rotate(pil_img, 180))
  tensor_270 = base_transform(TF.rotate(pil_img, 270))

  batch = torch.stack([tensor_0, tensor_90, tensor_180, tensor_270]).to(DEVICE)
  with torch.no_grad():
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).mean(dim=0)
  return probs.cpu().numpy()


# ==================== 4. 统计与指标输出 ====================
tier_map = {"Tier0": 0, "Tier1": 1, "Tier2": 2}
class_correct = [0, 0, 0]
class_total = [0, 0, 0]
correct, total = 0, 0

for tier_name, label in tier_map.items():
  folder = os.path.join(TEST_DIR, tier_name)
  if not os.path.exists(folder):
    continue
  for fname in os.listdir(folder):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
      continue
    img = Image.open(os.path.join(folder, fname)).convert("RGB")
    probs = predict_with_4tta(img)
    pred = int(probs.argmax())

    class_total[label] += 1
    total += 1
    if pred == label:
      correct += 1
      class_correct[label] += 1

print("\n" + "=" * 50)
print(f"Top-1 准确率: {correct / total * 100:.2f}% ({correct}/{total})")
for i in range(3):
  recall = (
      class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0.0
  )
  print(
      f"Tier {i} 召回率: {recall:.1f}% ({class_correct[i]}/{class_total[i]})"
  )
print("=" * 50)
4. 边缘端部署与落地规范 (Raspberry Pi 4B)框架选型：纯 CPU 边缘推理，采用 Tencent NCNN 框架（开启 ARM NEON 四线程及 FP16 半精度）。模型转换工具：必须使用 pnnx 直转（严禁走过时的 onnx2ncnn 网页工具，否则 LayerNorm 与 GELU 算子会报错）。# 转换命令
pnnx convnext.pt inputshape=[1,3,384,384]
输出节点映射：输入节点固定为 in0，输出节点固定为 out0。目标性能：树莓派 4B 单帧前向推理耗时控制在 $1.2\text{ s} \sim 1.8\text{ s}$。