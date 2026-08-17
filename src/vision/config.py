"""
文件作用：读取、校验并保存 YAML 实验配置。
File purpose: load, validate, override, and persist YAML experiment configs.

所有训练、评估和演示入口都使用同一份 ExperimentConfig，避免旧版脚本中
路径、输入尺寸、类别数和超参数分别写死而造成的不一致。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """数据路径与 DataLoader 配置。/ Dataset paths and loader settings."""

    raw_dir: str
    artifacts_dir: str
    val_ratio: float = 0.1  # 官方训练集划给 validation 的比例。
    num_workers: int = 4  # DataLoader 并行进程数；Windows 可按机器情况调整。
    batch_size: int = 32
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型结构配置。/ Model architecture configuration."""

    name: str
    num_classes: int = 200
    pretrained: bool = True
    attention_pooling: bool = False
    dropout: float = 0.3
    hidden_dim: int = 1024
    custom_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])


@dataclass
class PreprocessConfig:
    """训练与推理图像预处理配置。/ Image preprocessing configuration."""

    image_size: int = 224
    resize_size: int = 256
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    random_resized_crop: bool = True
    crop_scale: list[float] = field(default_factory=lambda: [0.6, 1.0])
    horizontal_flip: float = 0.5
    random_rotation: float = 0.0
    color_jitter: float = 0.0
    randaugment_ops: int = 0
    randaugment_magnitude: int = 9
    random_erasing: float = 0.0


@dataclass
class TrainingConfig:
    """优化器、训练轮数与模型选择配置。/ Optimisation and training settings."""

    epochs: int = 100
    seed: int = 42
    optimizer: str = "adamw"
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.02
    freeze_backbone_epochs: int = 0
    amp: bool = True
    ema: bool = False
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    selection_metric: str = "accuracy"  # 只能使用 validation 指标选模型。
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    """统一评估与 TTA 配置。/ Unified evaluation and TTA settings."""

    tta_horizontal_flip: bool = False
    top_k: int = 3


@dataclass
class OutputConfig:
    """实验输出目录与名称。/ Experiment output location and run name."""

    root_dir: str
    experiment_name: str


@dataclass
class ExperimentConfig:
    """完整实验配置，供训练、评估和 Demo 共享。/ Complete experiment contract."""

    data: DataConfig
    model: ModelConfig
    preprocess: PreprocessConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        """转换成可序列化字典，用于 checkpoint 和 resolved config。"""

        return asdict(self)


def _deep_update(target: dict[str, Any], path: str, value: Any) -> None:
    """应用 `section.key=value` 命令行覆盖。/ Apply a dotted config override."""

    keys = path.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"Unknown configuration path: {path}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise KeyError(f"Unknown configuration key: {path}")
    cursor[keys[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    """将命令行字符串解析为键和值，值遵循 YAML 类型规则。"""

    if "=" not in raw:
        raise ValueError(f"Override must use key=value syntax: {raw}")
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def _validate_config(config: ExperimentConfig) -> None:
    """执行关键配置约束检查，尽早阻止无效实验。"""

    if not 0.0 < config.data.val_ratio < 1.0:
        raise ValueError("data.val_ratio must be between 0 and 1")
    if config.model.num_classes <= 1:
        raise ValueError("model.num_classes must be greater than 1")
    if config.preprocess.image_size <= 0:
        raise ValueError("preprocess.image_size must be positive")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if config.training.selection_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("selection_metric must be accuracy or macro_f1")


def load_experiment_config(
    config_path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """读取 YAML 并构建类型化配置。/ Load YAML and build a typed configuration."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = copy.deepcopy(raw)
    for override in overrides or []:
        key, value = _parse_override(override)
        _deep_update(raw, key, value)

    required = {"data", "model", "preprocess", "training", "evaluation", "output"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"Missing config sections: {sorted(missing)}")

    config = ExperimentConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        preprocess=PreprocessConfig(**raw["preprocess"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output=OutputConfig(**raw["output"]),
    )
    _validate_config(config)
    return config


def save_resolved_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """保存本次运行最终生效的配置。/ Save the exact effective run config."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)

