"""
文件作用：统一计算细粒度分类指标与类别混淆分析。
File purpose: compute a consistent set of classification metrics and errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationMetrics:
    """一次评估的主要量化结果。/ Main scalar classification metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    top_k_accuracy: float
    num_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    top_k: int = 3,
) -> ClassificationMetrics:
    """计算 Accuracy、Macro P/R/F1、Weighted F1 与 Top-k Accuracy。"""

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    effective_k = min(top_k, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -effective_k, axis=1)[:, -effective_k:]
    top_k_accuracy = float(np.mean([target in indices for target, indices in zip(y_true, top_indices)]))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        top_k_accuracy=top_k_accuracy,
        num_samples=int(len(y_true)),
    )


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> list[dict[str, float | int | str]]:
    """为全部 200 类计算 Precision、Recall、F1 与 support。"""

    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return [
        {
            "class_index": index,
            "class_name": class_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    ]


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """强制生成固定 num_classes x num_classes 的混淆矩阵。"""

    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def find_top_confusion_pairs(
    matrix: np.ndarray, class_names: list[str], limit: int = 10
) -> list[dict[str, int | str]]:
    """找出除对角线外出现次数最多的类别混淆对。"""

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat_indices = np.argsort(off_diagonal.ravel())[::-1]
    pairs: list[dict[str, int | str]] = []
    for flat_index in flat_indices:
        true_index, predicted_index = np.unravel_index(flat_index, off_diagonal.shape)
        count = int(off_diagonal[true_index, predicted_index])
        if count <= 0 or len(pairs) >= limit:
            break
        pairs.append(
            {
                "true_class": class_names[true_index],
                "predicted_class": class_names[predicted_index],
                "count": count,
            }
        )
    return pairs

