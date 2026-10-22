# -*- coding: utf-8 -*-
import json
import pytest

from fabric_inspection.config import load_thresholds


def test_defaults():
    thresholds = load_thresholds()
    assert thresholds.cotton_gate == 0.20
    assert thresholds.focus == 75.0
    assert thresholds.focus_required is False


def test_load_and_override(tmp_path):
    path = tmp_path / "inspection.json"
    path.write_text(json.dumps({"cotton_gate": 0.0, "focus_required": True}),
                    encoding="utf-8")
    thresholds = load_thresholds(path)
    assert thresholds.cotton_gate == 0.0
    assert thresholds.focus_required is True
    assert thresholds.fiber_confidence == 0.50


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)
