# -*- coding: utf-8 -*-
"""级联判定：棉概率门控、三档判定与交叉校验。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np

from .labels import COTTON_TIERS, cotton_business, cotton_index, fiber_business


def softmax(logits: Sequence[float]) -> np.ndarray:
    """数值稳定的 softmax。已经是概率向量时原样归一化。"""
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("logits 不能为空")
    if np.all(values >= -1e-5) and abs(float(values.sum()) - 1.0) < 1e-3:
        return values / max(float(values.sum()), 1e-12)
    values = values - float(values.max())
    exp = np.exp(values)
    return exp / max(float(exp.sum()), 1e-12)


@dataclass
class Thresholds:
    """所有阈值集中管理，便于现场标定。"""

    cotton_gate: float = 0.20
    cotton_confidence: float = 0.46
    cotton_margin: float = 0.0
    fiber_confidence: float = 0.50
    focus: float = 75.0
    focus_required: bool = False


@dataclass
class InspectionResult:
    cotton_tier: str
    cotton_raw: str
    cotton_confidence: float
    cotton_margin: float
    cotton_probabilities: list[float]
    cotton_gate_probability: float
    cotton_entered: bool
    pure_fiber: str
    fiber_raw: str
    fiber_confidence: float
    fiber_probabilities: list[float]
    cross_check: str
    focus: float
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(cotton_probs, fiber_probs, fiber_names, thresholds: Thresholds,
           focus: float | None = None) -> InspectionResult:
    """执行一次完整判定。

    cotton_probs 为 None 表示未运行棉含量模型；函数会根据 P(棉) 自行决定
    是否需要它。返回结构可直接序列化为 JSON。
    """
    fiber_p = softmax(fiber_probs)
    if len(fiber_p) != len(fiber_names):
        raise ValueError("fiber_probs 长度与 fiber_names 不一致")

    index = cotton_index(fiber_names)
    gate_probability = float(fiber_p[index]) if index is not None else 0.0
    entered = gate_probability >= thresholds.cotton_gate

    fiber_top = int(np.argmax(fiber_p))
    fiber_raw = fiber_names[fiber_top]
    fiber_conf = float(fiber_p[fiber_top])
    pure_fiber = fiber_business(fiber_raw)
    if fiber_conf < thresholds.fiber_confidence:
        pure_fiber = "其他"

    if not entered:
        cotton_tier = "不适用"
        cotton_raw = "未运行"
        cotton_conf = 0.0
        cotton_margin = 0.0
        cotton_p = [0.0, 0.0, 0.0]
        cotton_valid = False
    else:
        if cotton_probs is None:
            raise ValueError("P(棉) 达到门限，但未提供 cotton_probs")
        cotton_p = softmax(cotton_probs)
        if len(cotton_p) != len(COTTON_TIERS):
            raise ValueError("cotton_probs 必须是三分类输出")
        order = np.sort(cotton_p)
        cotton_margin = float(order[-1] - order[-2])
        top = int(np.argmax(cotton_p))
        cotton_raw = COTTON_TIERS[top]
        cotton_conf = float(cotton_p[top])
        cotton_tier = cotton_business(cotton_raw)
        cotton_valid = (cotton_conf >= thresholds.cotton_confidence
                        and cotton_margin >= thresholds.cotton_margin)
        if not cotton_valid:
            cotton_tier = "待定"

    passed = entered and (
        (cotton_tier == "低棉" and pure_fiber == "其他")
        or (cotton_tier == "高棉" and pure_fiber == "纯棉")
    )
    if thresholds.focus_required and focus is not None and focus < thresholds.focus:
        passed = False

    return InspectionResult(
        cotton_tier=cotton_tier,
        cotton_raw=cotton_raw,
        cotton_confidence=round(cotton_conf, 5),
        cotton_margin=round(cotton_margin, 5),
        cotton_probabilities=[round(float(value), 5) for value in cotton_p],
        cotton_gate_probability=round(gate_probability, 5),
        cotton_entered=entered,
        pure_fiber=pure_fiber,
        fiber_raw=fiber_raw,
        fiber_confidence=round(fiber_conf, 5),
        fiber_probabilities=[round(float(value), 5) for value in fiber_p],
        cross_check="通过" if passed else "复核",
        focus=round(float(focus), 2) if focus is not None else 0.0,
        thresholds=asdict(thresholds),
    )
