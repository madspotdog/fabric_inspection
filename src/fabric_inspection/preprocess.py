# -*- coding: utf-8 -*-
"""图像预处理：中心 ROI、缩放、归一化和清晰度评估。"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def center_square(image: np.ndarray, side: int | None = None) -> np.ndarray:
    """取图像中心正方形区域。

    side 为 None 时取短边；否则取 min(高, 宽, side)。
    返回的是视图副本，避免后续原地修改影响原图。
    """
    if image.ndim not in (2, 3):
        raise ValueError("image 必须是 HxW 或 HxWxC 数组")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正数")
    target = min(height, width) if side is None else min(height, width, side)
    top = (height - target) // 2
    left = (width - target) // 2
    return image[top:top + target, left:left + target].copy()


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """把图像缩放到 size×size。

    优先使用 OpenCV 的 INTER_AREA；没有 OpenCV 时退化为最近邻，
    后者只用于无依赖的单元测试，正式推理应安装 opencv。
    """
    if size <= 0:
        raise ValueError("size 必须为正数")
    try:
        import cv2
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        height, width = image.shape[:2]
        rows = (np.arange(size) * height / size).astype(np.int32)
        cols = (np.arange(size) * width / size).astype(np.int32)
        return image[rows][:, cols]


def to_float_rgb(image: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float32，范围 0~1。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须是 HxWx3")
    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    return rgb


def imagenet_normalize(rgb: np.ndarray) -> np.ndarray:
    """按 ImageNet 均值/方差归一化，返回 HWC float32。"""
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def plain_normalize(rgb: np.ndarray) -> np.ndarray:
    """仅做 0~1 归一化，供 YOLO 系模型使用。"""
    return rgb.astype(np.float32)


def focus_score(image: np.ndarray) -> float:
    """用拉普拉斯方差衡量清晰度。

    数值越大表示边缘越锐利。不同相机/倍率下阈值需要重新标定，
    默认参考值为 75。
    """
    gray = image
    if image.ndim == 3:
        # BGR -> 灰度，使用整数权重近似
        gray = (image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587
                + image[:, :, 2] * 0.299)
    gray = gray.astype(np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    height, width = gray.shape
    result = np.zeros((height - 2, width - 2), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            result += kernel[row, col] * gray[row:row + height - 2,
                                               col:col + width - 2]
    return float(result.var())
