# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fabric_inspection.preprocess import (
    center_square, focus_score, imagenet_normalize, resize_square, to_float_rgb,
)


def test_center_square_uses_short_side():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = center_square(image)
    assert crop.shape == (720, 720, 3)


def test_center_square_respects_side():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = center_square(image, 660)
    assert crop.shape == (660, 660, 3)


def test_center_square_rejects_invalid():
    with pytest.raises(ValueError):
        center_square(np.zeros((0, 10, 3), dtype=np.uint8))


def test_resize_square():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_square(image, 32).shape == (32, 32, 3)


def test_rgb_and_normalize_range():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    rgb = to_float_rgb(image)
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    normalized = imagenet_normalize(rgb)
    assert normalized.shape == (8, 8, 3)
    assert normalized.dtype == np.float32


def test_focus_score_sharp_vs_flat():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    checker = np.repeat(checker[:, :, None], 3, axis=2)
    assert focus_score(flat) == 0.0
    assert focus_score(checker) > 1000
