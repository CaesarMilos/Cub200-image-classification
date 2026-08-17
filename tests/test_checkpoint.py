"""
文件作用：验证自描述 checkpoint 能恢复结构、类别映射与模型权重。
File purpose: test the v2 checkpoint round trip.
"""

import pytest

torch = pytest.importorskip("torch")

from src.vision.checkpoint import restore_model_from_checkpoint, save_checkpoint
from src.vision.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    TrainingConfig,
)
from src.vision.data import ClassMap
from src.vision.models import build_model


def test_checkpoint_round_trip(tmp_path) -> None:
    """小型 Custom CNN checkpoint 应严格恢复。"""

    config = ExperimentConfig(
        data=DataConfig(raw_dir="raw", artifacts_dir="artifacts"),
        model=ModelConfig(
            name="custom_cnn",
            num_classes=2,
            pretrained=False,
            custom_channels=[8, 16, 32, 64],
        ),
        preprocess=PreprocessConfig(image_size=64, resize_size=72),
        training=TrainingConfig(epochs=1, amp=False),
        evaluation=EvaluationConfig(top_k=2),
        output=OutputConfig(root_dir="outputs", experiment_name="test"),
    )
    class_map = ClassMap({1: "001.A", 2: "002.B"})
    model = build_model(config.model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        config=config,
        class_map=class_map,
        raw_state_dict=model.state_dict(),
        ema_state_dict=None,
        selected_weights="raw",
        epoch=1,
        validation_metrics={"accuracy": 0.5, "macro_f1": 0.4},
        # Exercise PyTorch 2.6 compatibility for historical TorchVersion metadata.
        environment={"torch": torch.__version__},
    )
    restored, restored_map, preprocess, metadata = restore_model_from_checkpoint(
        path, torch.device("cpu")
    )
    assert type(restored) is type(model)
    assert restored_map.class_to_idx == class_map.class_to_idx
    assert preprocess.image_size == 64
    assert metadata["selected_weights"] == "raw"
