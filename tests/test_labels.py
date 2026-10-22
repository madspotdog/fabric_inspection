# -*- coding: utf-8 -*-
from fabric_inspection.labels import cotton_business, cotton_index, fiber_business


def test_cotton_business_aliases():
    assert cotton_business("Tier0_Low") == "低棉"
    assert cotton_business("Tier1_Mid") == "中棉"
    assert cotton_business("Tier2_High") == "高棉"
    assert cotton_business("未知") == "其他"


def test_fiber_business_mapping():
    assert fiber_business("棉") == "纯棉"
    assert fiber_business("羊毛") == "纯羊毛"
    assert fiber_business("涤纶") == "纯涤纶"
    assert fiber_business("丝") == "其他"
    assert fiber_business("亚麻") == "其他"


def test_cotton_index():
    names = ["丝", "亚麻", "其他", "棉", "涤纶", "羊毛"]
    assert cotton_index(names) == 3
    assert cotton_index(["棉", "涤纶"]) == 0
    assert cotton_index(["丝"]) is None
