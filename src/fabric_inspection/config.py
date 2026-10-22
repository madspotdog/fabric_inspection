# -*- coding: utf-8 -*-
"""配置加载与校验。"""
from __future__ import annotations

import json
from pathlib import Path

from .decision import Thresholds

DEFAULTS = {
    "cotton_gate": 0.20,
    "cotton_confidence": 0.46,
    "cotton_margin": 0.0,
    "fiber_confidence": 0.50,
    "focus": 75.0,
    "focus_required": False,
    "cotton_input": 384,
    "fiber_input": 288,
    "roi": 660,
}


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """从 JSON 读取阈值；字段缺失时使用默认值。"""
    data = dict(DEFAULTS)
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(payload) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        data.update(payload)
    return Thresholds(
        cotton_gate=float(data["cotton_gate"]),
        cotton_confidence=float(data["cotton_confidence"]),
        cotton_margin=float(data["cotton_margin"]),
        fiber_confidence=float(data["fiber_confidence"]),
        focus=float(data["focus"]),
        focus_required=bool(data["focus_required"]),
    )
