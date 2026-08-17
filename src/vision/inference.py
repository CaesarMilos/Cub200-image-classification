"""
文件作用：提供面向单图 Demo 的 checkpoint 加载和 Top-k 推理接口。
File purpose: provide checkpoint-backed single-image inference for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .checkpoint import restore_model_from_checkpoint
from .data import ClassMap, build_transform


@dataclass
class ImagePrediction:
    """单图 Top-k 预测。/ Top-k prediction result for one image."""

    labels: list[str]
    scores: list[float]
    predicted_index: int


class Predictor:
    """封装模型、类别映射和 checkpoint 定义的预处理。"""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.model, self.class_map, preprocess, self.metadata = restore_model_from_checkpoint(
            checkpoint_path, self.device
        )
        self.transform = build_transform(preprocess, training=False)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> ImagePrediction:
        """对 PIL RGB 图像输出 Top-k 类别与置信度。"""

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)
        scores, indices = probabilities.topk(min(top_k, probabilities.shape[1]), dim=1)
        index_values = indices[0].cpu().tolist()
        return ImagePrediction(
            labels=[self.class_map.idx_to_class[int(index)] for index in index_values],
            scores=[float(value) for value in scores[0].cpu().tolist()],
            predicted_index=int(index_values[0]),
        )

