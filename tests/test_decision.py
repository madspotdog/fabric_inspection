# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fabric_inspection.decision import Thresholds, decide, softmax

FIBERS = ["丝", "亚麻", "其他", "棉", "涤纶", "羊毛"]


def fiber_with(cotton_probability: float, top_name: str = "棉"):
    """构造以指定类别为最高分的六类概率。"""
    probabilities = {name: 0.0 for name in FIBERS}
    if top_name == "棉":
        probabilities["棉"] = max(cotton_probability, 0.7)
        share = (1.0 - probabilities["棉"]) / (len(FIBERS) - 1)
        for name in FIBERS:
            if name != "棉":
                probabilities[name] = share
    else:
        probabilities[top_name] = 0.7
        probabilities["棉"] = cotton_probability
        rest = [name for name in FIBERS if name not in (top_name, "棉")]
        share = (1.0 - 0.7 - cotton_probability) / max(1, len(rest))
        for name in rest:
            probabilities[name] = share
    return [probabilities[name] for name in FIBERS]


def test_softmax_stable():
    result = softmax([1000, 1000, 1000])
    assert abs(float(result.sum()) - 1.0) < 1e-6
    assert np.allclose(result, [1 / 3] * 3)


def test_softmax_accepts_probabilities():
    result = softmax([0.2, 0.3, 0.5])
    assert float(result.sum()) == pytest.approx(1.0)


def test_gate_skips_cotton_model():
    result = decide(None, fiber_with(0.05, "涤纶"), FIBERS, Thresholds())
    assert result.cotton_entered is False
    assert result.cotton_tier == "不适用"
    assert result.pure_fiber == "纯涤纶"
    assert result.cross_check == "复核"


def test_low_cotton_and_other_passes():
    result = decide([0.9, 0.05, 0.05], fiber_with(0.3, "其他"), FIBERS, Thresholds())
    assert result.cotton_tier == "低棉"
    assert result.pure_fiber == "其他"
    assert result.cross_check == "通过"


def test_high_cotton_and_cotton_passes():
    result = decide([0.05, 0.05, 0.9], fiber_with(0.8, "棉"), FIBERS, Thresholds())
    assert result.cotton_tier == "高棉"
    assert result.pure_fiber == "纯棉"
    assert result.cross_check == "通过"


def test_mid_cotton_is_reviewed():
    result = decide([0.05, 0.9, 0.05], fiber_with(0.8, "棉"), FIBERS, Thresholds())
    assert result.cotton_tier == "中棉"
    assert result.cross_check == "复核"


def test_low_confidence_becomes_pending():
    result = decide([0.36, 0.34, 0.30], fiber_with(0.8, "棉"), FIBERS,
                    Thresholds(cotton_confidence=0.46))
    assert result.cotton_tier == "待定"
    assert result.cross_check == "复核"


def test_margin_rule():
    result = decide([0.40, 0.38, 0.22], fiber_with(0.8, "棉"), FIBERS,
                    Thresholds(cotton_confidence=0.30, cotton_margin=0.10))
    assert result.cotton_tier == "待定"


def test_low_fiber_confidence_becomes_other():
    probabilities = [0.30, 0.20, 0.20, 0.20, 0.05, 0.05]
    result = decide([0.9, 0.05, 0.05], probabilities, FIBERS, Thresholds())
    assert result.pure_fiber == "其他"


def test_focus_requirement_blocks_pass():
    result = decide([0.9, 0.05, 0.05], fiber_with(0.3, "其他"), FIBERS,
                    Thresholds(focus_required=True, focus=75), focus=10)
    assert result.cross_check == "复核"


def test_missing_cotton_probs_raises():
    with pytest.raises(ValueError):
        decide(None, fiber_with(0.9, "棉"), FIBERS, Thresholds())
