# -*- coding: utf-8 -*-
"""NCNN 推理封装与双模型流水线。

该模块与具体操作系统无关，树莓派、x86 或其他 Linux 开发板均可使用，
只要求安装 ncnn、numpy、opencv。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .decision import InspectionResult, Thresholds, decide
from .preprocess import center_square, focus_score, resize_square, to_float_rgb


class NcnnModel:
    """单个 NCNN 分类模型。

    imagenet=True 时使用 ImageNet 均值/方差，否则只做 0~1 归一化。
    """

    def __init__(self, param, binary, names, input_size: int,
                 threads: int = 2, imagenet: bool = True, fp16: bool = True):
        import ncnn

        self.ncnn = ncnn
        self.names = tuple(names)
        self.input_size = int(input_size)
        self.imagenet = bool(imagenet)
        self.net = ncnn.Net()
        self.net.opt.num_threads = int(threads)
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = bool(fp16)
        self.net.opt.use_fp16_storage = bool(fp16)
        self.net.opt.use_packing_layout = True
        if self.net.load_param(str(param)) != 0:
            raise RuntimeError(f"NCNN 结构加载失败: {param}")
        if self.net.load_model(str(binary)) != 0:
            raise RuntimeError(f"NCNN 权重加载失败: {binary}")

    def _mat(self, bgr: np.ndarray):
        crop = center_square(bgr, self.input_size)
        crop = resize_square(crop, self.input_size)
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        mat = self.ncnn.Mat.from_pixels(
            rgb.tobytes(), self.ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_size, self.input_size)
        if self.imagenet:
            mat.substract_mean_normalize(
                [0.485 * 255, 0.456 * 255, 0.406 * 255],
                [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)])
        else:
            mat.substract_mean_normalize([0, 0, 0], [1 / 255, 1 / 255, 1 / 255])
        return mat

    def predict(self, bgr: np.ndarray, tta: bool = False) -> np.ndarray:
        """返回类别概率向量。tta=True 时对 0/90/180/270 度取平均。"""
        frames = [bgr] if not tta else [np.rot90(bgr, k).copy() for k in range(4)]
        probabilities = []
        for frame in frames:
            extractor = self.net.create_extractor()
            if extractor.input("in0", self._mat(frame)) != 0:
                raise RuntimeError("NCNN 输入节点 in0 写入失败")
            code, output = extractor.extract("out0")
            if code != 0:
                raise RuntimeError(f"NCNN 推理失败: {code}")
            values = np.asarray(output, dtype=np.float32).reshape(-1)
            probabilities.append(_softmax(values))
        return np.mean(probabilities, axis=0)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - float(values.max())
    exp = np.exp(values)
    return exp / max(float(exp.sum()), 1e-12)


class InspectionPipeline:
    """级联双模型流水线：纯面料门控 -> 棉含量 -> 交叉校验。"""

    def __init__(self, cotton: NcnnModel, fiber: NcnnModel,
                 thresholds: Thresholds | None = None):
        self.cotton = cotton
        self.fiber = fiber
        self.thresholds = thresholds or Thresholds()

    def inspect(self, bgr: np.ndarray, tta: bool = False) -> InspectionResult:
        fiber_probs = self.fiber.predict(bgr, tta=tta)
        focus = focus_score(center_square(bgr, 660))
        from .labels import cotton_index
        index = cotton_index(self.fiber.names)
        gate = float(fiber_probs[index]) if index is not None else 0.0
        cotton_probs = None
        if gate >= self.thresholds.cotton_gate:
            cotton_probs = self.cotton.predict(bgr, tta=tta)
        return decide(cotton_probs, fiber_probs, self.fiber.names,
                      self.thresholds, focus=focus)

    @classmethod
    def from_directory(cls, model_dir, threads: int = 2, fp16: bool = True,
                       thresholds: Thresholds | None = None):
        """从包含 cotton_tier/pure_fiber 的模型目录构建流水线。"""
        import json

        model_dir = Path(model_dir)
        cotton_dir = model_dir / "cotton_tier"
        fiber_dir = model_dir / "pure_fiber"
        cotton_names = json.loads((cotton_dir / "labels.json").read_text(encoding="utf-8"))
        fiber_names = json.loads((fiber_dir / "labels.json").read_text(encoding="utf-8"))
        cotton = NcnnModel(cotton_dir / "model.ncnn.param",
                           cotton_dir / "model.ncnn.bin",
                           cotton_names, 384, threads, True, fp16)
        fiber = NcnnModel(fiber_dir / "model.ncnn.param",
                          fiber_dir / "model.ncnn.bin",
                          fiber_names, 288, threads, False, fp16)
        return cls(cotton, fiber, thresholds)
