"""
文件作用：执行统一批量推理、计算指标并保存结构化结果。
File purpose: run inference, compute metrics, and persist auditable evaluation outputs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClassMap
from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    find_top_confusion_pairs,
)


@dataclass
class PredictionRecord:
    """单张图片的可审查预测结果。/ Auditable per-image prediction record."""

    image_path: str
    true_index: int
    true_class: str
    predicted_index: int
    predicted_class: str
    confidence: float
    top_k_classes: str
    top_k_scores: str
    correct: bool


@dataclass
class EvaluationResult:
    """完整评估结果，包括标量指标、混淆矩阵与逐图预测。"""

    metrics: ClassificationMetrics
    confusion_matrix: np.ndarray
    per_class_metrics: list[dict[str, object]]
    top_confusion_pairs: list[dict[str, object]]
    predictions: list[PredictionRecord]


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_map: ClassMap,
    top_k: int = 3,
    tta_horizontal_flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PredictionRecord]]:
    """对一个有标签 DataLoader 推理；可选水平翻转 TTA。"""

    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_records: list[PredictionRecord] = []

    for images, targets, paths in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if tta_horizontal_flip:
            flipped_logits = model(torch.flip(images, dims=[3]))
            logits = (logits + flipped_logits) / 2.0
        probabilities = torch.softmax(logits, dim=1)
        effective_k = min(top_k, probabilities.shape[1])
        scores, indices = probabilities.topk(effective_k, dim=1)
        predictions = indices[:, 0]

        true_np = targets.numpy()
        pred_np = predictions.cpu().numpy()
        prob_np = probabilities.cpu().numpy()
        score_np = scores.cpu().numpy()
        index_np = indices.cpu().numpy()
        all_true.extend(true_np.tolist())
        all_pred.extend(pred_np.tolist())
        all_probabilities.extend(prob_np)

        for row, path in enumerate(paths):
            true_index = int(true_np[row])
            predicted_index = int(pred_np[row])
            top_classes = [class_map.idx_to_class[int(index)] for index in index_np[row]]
            all_records.append(
                PredictionRecord(
                    image_path=str(path),
                    true_index=true_index,
                    true_class=class_map.idx_to_class[true_index],
                    predicted_index=predicted_index,
                    predicted_class=class_map.idx_to_class[predicted_index],
                    confidence=float(score_np[row, 0]),
                    top_k_classes=" | ".join(top_classes),
                    top_k_scores=" | ".join(f"{value:.6f}" for value in score_np[row]),
                    correct=true_index == predicted_index,
                )
            )

    return (
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
        all_records,
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_map: ClassMap,
    top_k: int = 3,
    tta_horizontal_flip: bool = False,
) -> EvaluationResult:
    """完成推理并生成统一 EvaluationResult。"""

    y_true, y_pred, probabilities, records = predict_loader(
        model,
        loader,
        device,
        class_map,
        top_k=top_k,
        tta_horizontal_flip=tta_horizontal_flip,
    )
    class_names = [class_map.idx_to_class[index] for index in range(len(class_map.idx_to_class))]
    matrix = compute_confusion_matrix(y_true, y_pred, len(class_names))
    return EvaluationResult(
        metrics=compute_classification_metrics(y_true, y_pred, probabilities, top_k=top_k),
        confusion_matrix=matrix,
        per_class_metrics=compute_per_class_metrics(y_true, y_pred, class_names),
        top_confusion_pairs=find_top_confusion_pairs(matrix, class_names),
        predictions=records,
    )


def save_evaluation_result(result: EvaluationResult, output_dir: str | Path) -> None:
    """保存 JSON、CSV 与 NPY，保证最终指标可以复查。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics.to_dict(), handle, ensure_ascii=False, indent=2)
    np.save(output / "confusion_matrix.npy", result.confusion_matrix)

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "predictions.csv", [asdict(record) for record in result.predictions])
    write_csv(output / "per_class_metrics.csv", result.per_class_metrics)
    write_csv(output / "top_confusion_pairs.csv", result.top_confusion_pairs)


def print_metrics(metrics: ClassificationMetrics) -> None:
    """以适合终端 Demo 的格式显示核心指标。"""

    print("\nEvaluation results")
    print("-" * 40)
    print(f"Samples:         {metrics.num_samples}")
    print(f"Accuracy:        {metrics.accuracy:.4f}")
    print(f"Macro Precision: {metrics.macro_precision:.4f}")
    print(f"Macro Recall:    {metrics.macro_recall:.4f}")
    print(f"Macro F1:        {metrics.macro_f1:.4f}")
    print(f"Weighted F1:     {metrics.weighted_f1:.4f}")
    print(f"Top-k Accuracy:  {metrics.top_k_accuracy:.4f}")

