"""
文件作用：用最小输入验证 Custom CNN 前向传播输出形状。
File purpose: provide a quick model smoke test without the CUB dataset.
"""

import pytest

torch = pytest.importorskip("torch")

from src.vision.models import CustomFineGrainedCNN


def test_custom_cnn_forward_shape() -> None:
    """两张 64x64 RGB 图片应输出两行、五类 logits。"""

    model = CustomFineGrainedCNN(num_classes=5, channels=[8, 16, 32, 64], dropout=0.1)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 5)

