"""
文件作用：训练 ConvNeXt-Tiny baseline 或 Attention Pooling 主力模型。
File purpose: train the high-performance ConvNeXt experiment family.

该入口不再接受伪 ResNet 选项；ResNet 始终由 train_resnet.py 负责。
"""

from src.vision.cli import run_training_cli


if __name__ == "__main__":
    run_training_cli(
        "configs/convnext_attention.yaml", {"convnext_tiny", "convnext_attention"}
    )

