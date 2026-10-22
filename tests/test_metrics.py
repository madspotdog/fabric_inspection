# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fabric_inspection.metrics import (
    confusion_matrix, group_vote_accuracy, majority_vote, per_class_recall,
    top1_accuracy,
)

LABELS = ["低棉", "中棉", "高棉"]


def test_confusion_matrix_and_accuracy():
    matrix = confusion_matrix(["低棉", "低棉", "高棉"], ["低棉", "中棉", "高棉"], LABELS)
    assert matrix.tolist() == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert top1_accuracy(matrix) == pytest.approx(2 / 3)


def test_per_class_recall():
    matrix = np.array([[2, 0, 0], [1, 1, 0], [0, 0, 3]])
    assert per_class_recall(matrix) == pytest.approx([1.0, 0.5, 1.0])


def test_group_vote():
    groups = {
        "a": [np.array([0.6, 0.3, 0.1]), np.array([0.7, 0.2, 0.1])],
        "b": [np.array([0.1, 0.2, 0.7])],
    }
    truth = {"a": "低棉", "b": "高棉"}
    assert group_vote_accuracy(groups, truth, LABELS) == 1.0


def test_group_vote_partial():
    groups = {
        "a": [np.array([0.4, 0.35, 0.25])],
        "b": [np.array([0.1, 0.2, 0.7])],
    }
    truth = {"a": "高棉", "b": "高棉"}
    assert group_vote_accuracy(groups, truth, LABELS) == 0.5


def test_majority_vote_tie_keeps_first():
    assert majority_vote(["低棉", "高棉"]) == "低棉"
    assert majority_vote(["高棉", "低棉", "高棉"]) == "高棉"


def test_empty_vote_raises():
    with pytest.raises(ValueError):
        majority_vote([])
