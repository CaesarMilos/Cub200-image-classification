"""
文件作用：训练 ImageNet 预训练 ResNet-50 标准迁移学习基线。
File purpose: train the standard transfer-learning baseline.
"""

from src.vision.cli import run_training_cli


if __name__ == "__main__":
    run_training_cli("configs/resnet50.yaml", {"resnet50"})

