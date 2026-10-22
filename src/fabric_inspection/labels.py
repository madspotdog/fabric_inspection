# -*- coding: utf-8 -*-
"""类别定义与业务映射。"""
from __future__ import annotations

COTTON_TIERS = ("低棉", "中棉", "高棉")
FIBER_CLASSES = ("丝", "亚麻", "其他", "棉", "涤纶", "羊毛")
FIBER_BUSINESS = ("纯棉", "纯羊毛", "纯涤纶", "其他")

_TIER_ALIASES = {
    "Tier0_Low": "低棉", "低棉": "低棉",
    "Tier1_Mid": "中棉", "混纺": "中棉", "中棉": "中棉",
    "Tier2_High": "高棉", "高棉": "高棉",
}

_FIBER_ALIASES = {
    "棉": "纯棉", "纯棉": "纯棉",
    "羊毛": "纯羊毛", "纯羊毛": "纯羊毛",
    "涤纶": "纯涤纶", "纯涤纶": "纯涤纶",
    "丝": "其他", "亚麻": "其他", "其他": "其他",
}


def cotton_business(name: str) -> str:
    """把模型原始类别名转换为业务档位。未知类别返回“其他”。"""
    return _TIER_ALIASES.get(name, "其他")


def fiber_business(name: str) -> str:
    """把六类纤维名转换为业务四类。未知类别返回“其他”。"""
    return _FIBER_ALIASES.get(name, "其他")


def cotton_index(fiber_names) -> int | None:
    """返回纯面料模型中“棉”类别的索引；不存在时返回 None。"""
    for index, name in enumerate(fiber_names):
        if name == "棉":
            return index
    return None
