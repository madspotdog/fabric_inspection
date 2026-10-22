# -*- coding: utf-8 -*-
"""评估指标：混淆矩阵、召回率、整布表决。"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """按 labels 顺序生成混淆矩阵，行=真值，列=预测。"""
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        if truth not in index or pred not in index:
            raise ValueError(f"未知类别: {truth}, {pred}")
        matrix[index[truth], index[pred]] += 1
    return matrix


def per_class_recall(matrix: np.ndarray) -> list[float]:
    """逐类召回率；无样本的类别返回 0。"""
    result = []
    for i in range(matrix.shape[0]):
        total = int(matrix[i].sum())
        result.append(float(matrix[i, i] / total) if total else 0.0)
    return result


def top1_accuracy(matrix: np.ndarray) -> float:
    total = int(matrix.sum())
    return float(np.trace(matrix) / total) if total else 0.0


def group_vote_accuracy(groups: dict[str, list[np.ndarray]],
                        group_truth: dict[str, str],
                        labels) -> float:
    """整布表决：同一布料多视角概率取平均后再取最大类。"""
    if not groups:
        return 0.0
    index = {label: i for i, label in enumerate(labels)}
    correct = 0
    for group, probabilities in groups.items():
        mean = np.mean(np.asarray(probabilities, dtype=np.float32), axis=0)
        pred = labels[int(np.argmax(mean))]
        correct += int(pred == group_truth[group])
    return correct / len(groups)


def majority_vote(predictions) -> str:
    """离散标签的多数表决；票数相同时保留首次出现的标签。"""
    if not predictions:
        raise ValueError("predictions 不能为空")
    counts = Counter(predictions)
    best = max(counts.values())
    for label in predictions:
        if counts[label] == best:
            return label
    return predictions[0]


def accumulate(groups: dict[str, list[np.ndarray]], key: str,
               probabilities: np.ndarray) -> None:
    """把一次推理结果追加到指定布料的概率列表。"""
    groups.setdefault(key, []).append(np.asarray(probabilities, dtype=np.float32))
