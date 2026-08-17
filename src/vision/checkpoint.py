"""
文件作用：定义可复现 checkpoint 格式并负责模型恢复。
File purpose: save self-describing checkpoints and restore models safely.

新版 checkpoint 同时保存结构、预处理、类别映射、环境、raw/EMA 权重与验证指标，
修复旧版最佳 EMA 验证结果无法由提交 checkpoint 独立复现的问题。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, ModelConfig, PreprocessConfig
from .data import ClassMap
from .models import build_model


def save_checkpoint(
    path: str | Path,
    *,
    config: ExperimentConfig,
    class_map: ClassMap,
    raw_state_dict: dict[str, torch.Tensor],
    ema_state_dict: dict[str, torch.Tensor] | None,
    selected_weights: str,
    epoch: int,
    validation_metrics: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> None:
    """保存包含完整实验契约的 checkpoint。/ Save a self-contained checkpoint."""

    if selected_weights not in {"raw", "ema"}:
        raise ValueError("selected_weights must be raw or ema")
    if selected_weights == "ema" and ema_state_dict is None:
        raise ValueError("EMA checkpoint requires ema_state_dict")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "metadata": {
            "config": config.to_dict(),
            "class_map": class_map.to_dict(),
            "selected_weights": selected_weights,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "environment": environment or {},
        },
        "raw_model_state_dict": raw_state_dict,
        "ema_model_state_dict": ema_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
    }
    torch.save(payload, output)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """安全读取新版 checkpoint，并兼容 PyTorch 2.6 的 TorchVersion metadata。"""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as error:
        # PyTorch 2.6 treats torch.__version__ as a TorchVersion object. Older v2
        # checkpoints may persist it in environment metadata, so allow only this
        # trusted PyTorch metadata type while keeping weights_only protection.
        if "torch.torch_version.TorchVersion" not in str(error):
            raise
        from torch.torch_version import TorchVersion

        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError(
            "This is not a v2 self-describing checkpoint. Legacy state_dict files require migration."
        )
    return payload


def restore_model_from_checkpoint(
    path: str | Path,
    device: torch.device,
    weights_variant: str | None = None,
) -> tuple[torch.nn.Module, ClassMap, PreprocessConfig, dict[str, Any]]:
    """从 checkpoint 恢复模型、类别映射、transform 配置与 metadata。"""

    payload = load_checkpoint_payload(path, map_location=device)
    metadata = payload["metadata"]
    config_dict = metadata["config"]
    model_config = ModelConfig(**config_dict["model"])
    preprocess_config = PreprocessConfig(**config_dict["preprocess"])

    class_map_payload = metadata["class_map"]
    class_map = ClassMap.__new__(ClassMap)
    class_map.class_to_idx = {
        str(k): int(v) for k, v in class_map_payload["class_to_idx"].items()
    }
    class_map.idx_to_class = {
        int(k): str(v) for k, v in class_map_payload["idx_to_class"].items()
    }

    model = build_model(model_config)
    variant = weights_variant or metadata["selected_weights"]
    if variant == "ema":
        state_dict = payload.get("ema_model_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint does not contain EMA weights")
    elif variant == "raw":
        state_dict = payload["raw_model_state_dict"]
    else:
        raise ValueError("weights_variant must be raw or ema")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, class_map, preprocess_config, metadata
