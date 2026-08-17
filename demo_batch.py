"""
文件作用：对老师提供的任意有标签图片集执行课程现场评估。
File purpose: evaluate arbitrary labelled demo images without rebuilding class indices.

labels.csv 至少包含 `image,class_name` 两列；class_name 必须来自 checkpoint 类别映射。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.vision.checkpoint import restore_model_from_checkpoint
from src.vision.data import LabelledImageDataset, build_transform
from src.vision.evaluator import evaluate_loader, print_metrics, save_evaluation_result
from src.vision.visualization import save_confusion_matrix


def main() -> None:
    """加载任意 images + labels.csv 并输出逐图预测与课程要求的全部指标。"""

    parser = argparse.ArgumentParser(description="Batch demo for labelled CUB images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir", default="demo_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_map, preprocess, metadata = restore_model_from_checkpoint(
        args.checkpoint, device
    )
    dataset = LabelledImageDataset(
        args.images_dir,
        args.labels_csv,
        class_map,
        build_transform(preprocess, training=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    top_k = int(metadata["config"]["evaluation"]["top_k"])
    result = evaluate_loader(
        model, loader, device, class_map, top_k=top_k, tta_horizontal_flip=args.tta
    )
    output_dir = Path(args.output_dir)
    save_evaluation_result(result, output_dir)
    save_confusion_matrix(result.confusion_matrix, output_dir / "confusion_matrix.png")
    print_metrics(result.metrics)
    print("\nIndividual predictions")
    for record in result.predictions:
        status = "OK" if record.correct else "WRONG"
        print(
            f"{Path(record.image_path).name}: true={record.true_class} "
            f"pred={record.predicted_class} confidence={record.confidence:.4f} [{status}]"
        )
    print(f"Saved full demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()

