"""
文件作用：一键评估已经冻结的四组最佳 raw checkpoint，并生成最终 official test 结果。
File purpose: run one-click official test evaluation for all frozen experiment checkpoints.

该脚本不参与训练、调参或模型选择。它只读取各实验目录中时间最新的
`best_val_raw.pt`，并把正式指标保存到对应 run 目录中。Attention 模型额外运行一次
固定 checkpoint 的水平翻转 TTA，作为独立推理消融结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

from evaluate import main as evaluate_main


# 所有正式训练输出都位于工程根目录的 outputs/ 下。
OUTPUT_ROOT = Path("outputs")

# 顺序与 experiment_log.md 的最终对比表保持一致。
EXPERIMENTS = (
    "custom_cnn_v2",
    "resnet50_v2",
    "convnext_baseline_v2",
    "convnext_attention_v2",
)


def find_latest_raw_checkpoint(experiment_name: str) -> Path:
    """返回指定实验目录中最近一次运行的最佳 validation raw checkpoint。"""

    experiment_dir = OUTPUT_ROOT / experiment_name
    candidates = list(experiment_dir.glob("*/best_val_raw.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best_val_raw.pt found for {experiment_name}. "
            f"Expected under: {experiment_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_official_test(checkpoint_path: Path, output_name: str, use_tta: bool = False) -> None:
    """调用统一 evaluate.py，在 official test manifest 上评估一个冻结 checkpoint。"""

    output_dir = checkpoint_path.parent / output_name
    arguments = [
        "evaluate.py",
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "test",
        "--weights",
        "raw",
        "--output_dir",
        str(output_dir),
    ]
    if use_tta:
        arguments.append("--tta")

    previous_argv = sys.argv
    try:
        sys.argv = arguments
        evaluate_main()
    finally:
        sys.argv = previous_argv


def main() -> None:
    """依次执行四组模型的正式 test，并对 Attention 模型补充固定 TTA 推理。"""

    attention_checkpoint: Path | None = None
    for experiment_name in EXPERIMENTS:
        checkpoint = find_latest_raw_checkpoint(experiment_name)
        print(f"\n{'=' * 72}\nOfficial test: {experiment_name}\nCheckpoint: {checkpoint}")
        run_official_test(checkpoint, "final_test_raw")
        if experiment_name == "convnext_attention_v2":
            attention_checkpoint = checkpoint

    if attention_checkpoint is not None:
        print(f"\n{'=' * 72}\nOfficial test with TTA: convnext_attention_v2")
        run_official_test(attention_checkpoint, "final_test_tta", use_tta=True)

    print("\nAll official test evaluations are complete.")


if __name__ == "__main__":
    main()
