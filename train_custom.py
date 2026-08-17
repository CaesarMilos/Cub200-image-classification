"""
文件作用：训练从零设计的 Multi-Scale SE-Residual Custom CNN。
File purpose: train the coursework-required custom model from scratch.
"""

from src.vision.cli import run_training_cli


if __name__ == "__main__":
    run_training_cli("configs/custom_cnn.yaml", {"custom_cnn"})

