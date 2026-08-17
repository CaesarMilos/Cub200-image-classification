"""
文件作用：将最终 Attention 模型的 official test 原始结果转换为作品集展示图片。
File purpose: turn frozen official-test artifacts into portfolio-ready visualizations.

直接在 VS Code 打开本文件并点击右上角“运行 Python 文件”。本脚本不训练、不评估、
不修改 checkpoint；它只读取 final_test_raw/ 中已保存的 CSV/NPY 文件并生成 PNG。
"""

from __future__ import annotations

# Windows 上 PyTorch 与 NumPy/Matplotlib 可能各加载一份 Intel OpenMP runtime。
# 本脚本只进行冻结结果的离线可视化，允许共存以避免 OMP Error #15 直接中止进程。
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.vision.checkpoint import restore_model_from_checkpoint
from src.vision.data import build_transform
from src.vision.visualization import GradCAM, overlay_heatmap


# 关键参数：画廊各展示 3 张正确案例和 3 张高置信度错误案例，兼顾正反证据。
NUM_CORRECT_EXAMPLES = 3
NUM_ERROR_EXAMPLES = 3
# 关键参数：Grad-CAM 使用 2 个正确和 2 个错误案例，便于在报告中清晰阅读。
NUM_GRADCAM_EXAMPLES = 4


def find_latest_attention_evaluation(project_root: Path) -> Path:
    """定位 ConvNeXt Attention 最新一次已完成的 official-test 原始结果目录。"""

    candidates = list(
        (project_root / "outputs" / "convnext_attention_v2").glob("*/final_test_raw/metrics.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "未找到 Attention 的 final_test_raw/metrics.json。请先运行 evaluate_all_final.py。"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_raw_data_dir(project_root: Path) -> Path:
    """从 configs/data.yaml 读取 CUB-200 原始数据目录，用于加载预测案例图片。"""

    with (project_root / "configs" / "data.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw_dir = Path(config["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"configs/data.yaml 中的 raw_dir 不存在：{raw_dir}")
    return raw_dir


def short_class_name(class_name: str) -> str:
    """将官方类别名压缩为适合图表标签的短名称。"""

    return class_name.split(".", maxsplit=1)[-1].replace("_", " ")


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """绘制完整 200 类混淆矩阵，保留官方测试的全量审计证据。"""

    figure, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Number of samples")
    axis.set(
        title="ConvNeXt + Attention: Official Test Confusion Matrix",
        xlabel="Predicted class index",
        ylabel="True class index",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_top_confusion_pairs(rows: list[dict[str, str]], output_path: Path) -> None:
    """绘制测试集中出现次数最多的十个有方向类别混淆对。"""

    labels = [
        f"{short_class_name(row['true_class'])}  →  {short_class_name(row['predicted_class'])}"
        for row in rows
    ]
    counts = [int(row["count"]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(labels))
    axis.barh(positions, counts, color="#E45756")
    axis.set_yticks(positions, labels, fontsize=9)
    axis.invert_yaxis()
    axis.set(
        title="Top-10 Directional Confusion Pairs on Official Test Set",
        xlabel="Number of misclassified images",
    )
    axis.grid(axis="x", alpha=0.25)
    for position, count in zip(positions, counts):
        axis.text(count + 0.05, position, str(count), va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def choose_prediction_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """选取高置信度正确案例和高置信度错误案例，避免只展示有利样本。"""

    correct = sorted(
        (row for row in rows if row["correct"].lower() == "true"),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[:NUM_CORRECT_EXAMPLES]
    incorrect = sorted(
        (row for row in rows if row["correct"].lower() != "true"),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[:NUM_ERROR_EXAMPLES]
    selected = correct + incorrect
    if len(selected) != NUM_CORRECT_EXAMPLES + NUM_ERROR_EXAMPLES:
        raise ValueError("预测记录不足，无法生成完整案例画廊。")
    return selected


def plot_prediction_gallery(rows: list[dict[str, str]], raw_dir: Path, output_path: Path) -> None:
    """生成包含真实标签、预测标签、置信度与 Top-3 候选的六宫格预测案例。"""

    selected = choose_prediction_examples(rows)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    for index, (axis, row) in enumerate(zip(axes.flat, selected), start=1):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            # 兼容项目移动后的绝对路径：从 images/ 后的相对部分重新定位。
            try:
                relative_path = image_path.parts[image_path.parts.index("images") + 1 :]
                image_path = raw_dir / "images" / Path(*relative_path)
            except ValueError as error:
                raise FileNotFoundError(f"无法定位预测案例原图：{row['image_path']}") from error
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        is_correct = row["correct"].lower() == "true"
        status = "Correct" if is_correct else "Incorrect"
        color = "#2CA02C" if is_correct else "#D62728"
        true_name = short_class_name(row["true_class"])
        predicted_name = short_class_name(row["predicted_class"])
        top3 = row["top_k_classes"].replace(" | ", ", ")
        axis.set_title(
            f"{index}. {status}\nTrue: {true_name}\nPred: {predicted_name} ({float(row['confidence']) * 100:.1f}%)\nTop-3: {top3}",
            color=color,
            fontsize=8.5,
        )
        axis.axis("off")
    figure.suptitle("ConvNeXt + Attention: Official Test Prediction Examples", fontsize=15, y=0.99)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def resolve_image_path(recorded_path: str, raw_dir: Path) -> Path:
    """在工程移动后仍可从预测 CSV 中恢复原始 CUB 图片路径。"""

    image_path = Path(recorded_path)
    if image_path.exists():
        return image_path
    try:
        relative_path = image_path.parts[image_path.parts.index("images") + 1 :]
    except ValueError as error:
        raise FileNotFoundError(f"无法定位预测案例原图：{recorded_path}") from error
    recovered = raw_dir / "images" / Path(*relative_path)
    if not recovered.exists():
        raise FileNotFoundError(f"无法定位预测案例原图：{recorded_path}")
    return recovered


def select_gradcam_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """从预测记录选取两个正确和两个错误的高置信度 Grad-CAM 案例。"""

    correct = sorted(
        (row for row in rows if row["correct"].lower() == "true"),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[: NUM_GRADCAM_EXAMPLES // 2]
    incorrect = sorted(
        (row for row in rows if row["correct"].lower() != "true"),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[: NUM_GRADCAM_EXAMPLES // 2]
    return correct + incorrect


def plot_gradcam_gallery(
    rows: list[dict[str, str]], raw_dir: Path, checkpoint_path: Path, output_path: Path
) -> None:
    """恢复最终 raw checkpoint，并为正确/错误案例生成 Grad-CAM 原图和热力图对照。"""

    selected = select_gradcam_examples(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess, _ = restore_model_from_checkpoint(checkpoint_path, device, "raw")
    transform = build_transform(preprocess, training=False)
    gradcam = GradCAM(model)
    figure, axes = plt.subplots(len(selected), 2, figsize=(10, 4.2 * len(selected)))
    try:
        for row_index, row in enumerate(selected):
            image_path = resolve_image_path(row["image_path"], raw_dir)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            heatmap = gradcam.generate(input_tensor, class_index=int(row["predicted_index"]))
            overlay = overlay_heatmap(image, heatmap)
            is_correct = row["correct"].lower() == "true"
            status = "Correct" if is_correct else "Incorrect"
            color = "#2CA02C" if is_correct else "#D62728"
            true_name = short_class_name(row["true_class"])
            predicted_name = short_class_name(row["predicted_class"])
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(f"{status} | True: {true_name}", color=color, fontsize=10)
            axes[row_index, 1].imshow(overlay)
            axes[row_index, 1].set_title(
                f"Grad-CAM | Pred: {predicted_name} ({float(row['confidence']) * 100:.1f}%)",
                color=color,
                fontsize=10,
            )
            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")
    finally:
        gradcam.close()
    figure.suptitle("ConvNeXt + Attention: Grad-CAM on Official Test Examples", fontsize=15, y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取评估阶段生成的 CSV 表格。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """读取 frozen official-test artifacts 并生成四张可直接嵌入报告的 PNG 图片。"""

    project_root = Path(__file__).resolve().parent
    evaluation_dir = find_latest_attention_evaluation(project_root)
    figure_dir = project_root / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.load(evaluation_dir / "confusion_matrix.npy")
    confusion_rows = read_csv(evaluation_dir / "top_confusion_pairs.csv")
    prediction_rows = read_csv(evaluation_dir / "predictions.csv")
    raw_dir = load_raw_data_dir(project_root)

    outputs = {
        "confusion matrix": figure_dir / "attention_confusion_matrix.png",
        "top confusion pairs": figure_dir / "top_confusion_pairs.png",
        "prediction gallery": figure_dir / "prediction_gallery.png",
        "Grad-CAM gallery": figure_dir / "gradcam_gallery.png",
    }
    plot_confusion_matrix(matrix, outputs["confusion matrix"])
    plot_top_confusion_pairs(confusion_rows, outputs["top confusion pairs"])
    plot_prediction_gallery(prediction_rows, raw_dir, outputs["prediction gallery"])
    plot_gradcam_gallery(
        prediction_rows,
        raw_dir,
        evaluation_dir.parent / "best_val_raw.pt",
        outputs["Grad-CAM gallery"],
    )

    print(f"Source evaluation: {evaluation_dir.relative_to(project_root)}")
    for name, path in outputs.items():
        print(f"Saved {name}: {path.relative_to(project_root)}")
    print("Final visualization generation complete.")


if __name__ == "__main__":
    main()
