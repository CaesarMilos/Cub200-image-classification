"""
文件作用：将旧版纯 state_dict 包装成新版自描述 raw checkpoint。
File purpose: migrate legacy ResNet/ConvNeXt state dictionaries into the v2 format.

限制：旧文件没有保存 EMA shadow，因此只能迁移 raw 权重，无法恢复旧报告使用的 EMA 状态。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.vision.checkpoint import save_checkpoint
from src.vision.config import load_experiment_config
from src.vision.data import ClassMap
from src.vision.models import build_model
from src.vision.seed import collect_environment_info


def _load_legacy_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """读取旧 state_dict，兼容 `state_dict` 包装和 DataParallel 前缀。"""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Legacy checkpoint is not a state_dict mapping")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key.removeprefix("module.")
        # 旧版 AttentionPoolingHead 使用 attn/fc，新版使用更明确的名称。
        name = name.replace("head.attn.", "head.attention.")
        name = name.replace("head.fc.", "head.classifier.")
        cleaned[name] = value
    return cleaned


def main() -> None:
    """校验旧权重与配置结构匹配后，保存新版 raw checkpoint。"""

    parser = argparse.ArgumentParser(description="Migrate a legacy state_dict checkpoint")
    parser.add_argument("--legacy_checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--legacy_test_accuracy",
        type=float,
        default=None,
        help="Optional historical value for traceability; it is not treated as a reproduced metric.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    class_map = ClassMap.from_json(Path(config.data.artifacts_dir) / "class_to_idx.json")
    model = build_model(config.model)
    state_dict = _load_legacy_state_dict(args.legacy_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    metrics = {
        "accuracy": args.legacy_test_accuracy,
        "macro_f1": None,
        "status": "legacy_unverified",
    }
    save_checkpoint(
        args.output,
        config=config,
        class_map=class_map,
        raw_state_dict=model.state_dict(),
        ema_state_dict=None,
        selected_weights="raw",
        epoch=-1,
        validation_metrics=metrics,
        environment=collect_environment_info(),
    )
    print(f"Migrated legacy raw weights to: {args.output}")


if __name__ == "__main__":
    main()

