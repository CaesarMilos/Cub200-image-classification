"""
文件作用：生成训练曲线、混淆矩阵与 Grad-CAM 可解释性图像。
File purpose: create presentation-ready plots and Grad-CAM explanations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import get_gradcam_target_layer


def save_confusion_matrix(
    matrix: np.ndarray, output_path: str | Path, title: str = "Confusion Matrix"
) -> None:
    """保存适合 200 类任务的无文字热力图。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(title=title, xlabel="Predicted class index", ylabel="True class index")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    """从 history.csv 绘制 train/validation Accuracy 曲线。"""

    with Path(history_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    val_accuracy = [float(row["val_accuracy_raw"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_accuracy, label="Train accuracy")
    axis.plot(epochs, val_accuracy, label="Validation accuracy (raw)")
    axis.set(xlabel="Epoch", ylabel="Accuracy", title="Training history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


class GradCAM:
    """无需第三方依赖的通用 Grad-CAM 实现。/ Lightweight Grad-CAM implementation."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = get_gradcam_target_layer(model)
        self.forward_handle = target_layer.register_forward_hook(self._capture_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _capture_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """生成归一化到 [0,1] 的二维热力图。"""

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        target = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[:, target].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        """移除 hooks，避免重复创建 Demo 时累积回调。"""

        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """将 Grad-CAM 叠加到原图，返回用于 Gradio 展示的 RGB 图像。"""

    base = image.convert("RGB")
    resized_heatmap = Image.fromarray(np.uint8(heatmap * 255)).resize(base.size)
    colored = plt.get_cmap("jet")(np.asarray(resized_heatmap) / 255.0)[..., :3]
    colored_image = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored_image, alpha=alpha)

