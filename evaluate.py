"""
文件作用：加载自描述 checkpoint，在固定 val/test manifest 上生成正式评估结果。
File purpose: run the only authoritative validation/test evaluation path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.vision.checkpoint import restore_model_from_checkpoint
from src.vision.data import ManifestImageDataset, build_transform, load_manifest
from src.vision.evaluator import evaluate_loader, print_metrics, save_evaluation_result
from src.vision.visualization import save_confusion_matrix


def main() -> None:
    """恢复 checkpoint 契约并对指定固定 split 评估。"""

    parser = argparse.ArgumentParser(description="Evaluate a v2 CUB checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="Enable horizontal-flip TTA")
    parser.add_argument("--weights", choices=["raw", "ema"], default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_map, preprocess, metadata = restore_model_from_checkpoint(
        args.checkpoint, device, args.weights
    )
    config = metadata["config"]
    artifacts_dir = Path(config["data"]["artifacts_dir"])
    dataset = ManifestImageDataset(
        load_manifest(artifacts_dir / f"{args.split}_manifest.csv"),
        config["data"]["raw_dir"],
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
    result = evaluate_loader(
        model,
        loader,
        device,
        class_map,
        top_k=int(config["evaluation"]["top_k"]),
        tta_horizontal_flip=args.tta,
    )
    output_dir = Path(args.output_dir or f"evaluation_{args.split}")
    save_evaluation_result(result, output_dir)
    save_confusion_matrix(
        result.confusion_matrix,
        output_dir / "confusion_matrix.png",
        title=f"{args.split.title()} Confusion Matrix - Acc {result.metrics.accuracy:.2%}",
    )
    print_metrics(result.metrics)
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()

