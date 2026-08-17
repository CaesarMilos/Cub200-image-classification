"""
文件作用：汇总已完成训练的 history.csv，生成论文/作品集可用的训练与验证曲线。
File purpose: create presentation-ready training curves from completed experiment histories.

直接在 VS Code 打开本文件并点击右上角“运行 Python 文件”即可；无需重新训练模型。
图片默认保存到 results/figures/，不会修改 checkpoint、训练记录或数据集。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# 关键参数：实验目录名前缀和图中展示名称，必须与 configs 中的 experiment_name 保持对应。
EXPERIMENTS = {
    "custom_cnn_v2": "Custom CNN",
    "resnet50_transfer_v2": "ResNet-50 transfer",
    "convnext_baseline_v2": "ConvNeXt-Tiny baseline",
    "convnext_attention_v2": "ConvNeXt + Attention",
}


def find_latest_history(output_root: Path, experiment_name: str) -> Path:
    """在一个实验目录中定位最新一次正式运行的 history.csv。"""

    candidates = list((output_root / experiment_name).glob("*/history.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"未找到 {experiment_name} 的 history.csv。请确认该模型已训练完成且 outputs 文件夹保留。"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_history(history_path: Path) -> list[dict[str, str]]:
    """读取一个训练运行按 epoch 保存的指标记录。"""

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"训练记录为空：{history_path}")
    return rows


def plot_single_experiment(label: str, rows: list[dict[str, str]], output_path: Path) -> None:
    """绘制单个模型的训练准确率、验证准确率和验证 Macro-F1 曲线。"""

    epochs = [int(row["epoch"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) * 100 for row in rows]
    val_accuracy = [float(row["val_accuracy_raw"]) * 100 for row in rows]
    val_macro_f1 = [float(row["val_macro_f1_raw"]) * 100 for row in rows]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, train_accuracy, label="Train accuracy", color="#4C78A8", linewidth=2)
    axis.plot(epochs, val_accuracy, label="Validation accuracy", color="#F58518", linewidth=2)
    axis.plot(epochs, val_macro_f1, label="Validation Macro-F1", color="#54A24B", linewidth=2)
    axis.set(
        title=f"{label}: Training History",
        xlabel="Epoch",
        ylabel="Score (%)",
        ylim=(0, 102),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_validation_comparison(histories: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    """对比所有模型随 epoch 变化的验证准确率，便于展示模型改进趋势。"""

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, rows in histories.items():
        epochs = [int(row["epoch"]) for row in rows]
        validation_accuracy = [float(row["val_accuracy_raw"]) * 100 for row in rows]
        axis.plot(epochs, validation_accuracy, label=label, linewidth=2)

    axis.set(
        title="Validation Accuracy Comparison",
        xlabel="Epoch",
        ylabel="Validation Accuracy (%)",
        ylim=(0, 102),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """读取四个完成的实验并生成五张 PNG 曲线图。"""

    project_root = Path(__file__).resolve().parent
    output_root = project_root / "outputs"
    figure_root = project_root / "results" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list[dict[str, str]]] = {}
    print("Generating training curves...")
    for experiment_name, label in EXPERIMENTS.items():
        history_path = find_latest_history(output_root, experiment_name)
        rows = load_history(history_path)
        histories[label] = rows
        figure_path = figure_root / f"{experiment_name}_history.png"
        plot_single_experiment(label, rows, figure_path)
        print(f"Saved: {figure_path.relative_to(project_root)}")

    comparison_path = figure_root / "validation_accuracy_comparison.png"
    plot_validation_comparison(histories, comparison_path)
    print(f"Saved: {comparison_path.relative_to(project_root)}")
    print("Curve generation complete.")


if __name__ == "__main__":
    main()
