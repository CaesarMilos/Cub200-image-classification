"""
文件作用：为三个训练脚本提供一致的命令行解析和实验启动逻辑。
File purpose: share command-line parsing across all training entry points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .engine import run_training_experiment


def run_training_cli(default_config: str, expected_model: set[str]) -> None:
    """读取配置、应用 `--set` 覆盖并启动训练。"""

    parser = argparse.ArgumentParser(description="CUB-200-2011 training entry point")
    parser.add_argument(
        "--config", default=default_config, help="YAML experiment configuration path"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.epochs=2",
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.set)
    if config.model.name not in expected_model:
        raise ValueError(
            f"{Path(default_config).name} entry does not accept model.name={config.model.name}. "
            f"Expected one of {sorted(expected_model)}."
        )
    output_dir = run_training_experiment(config)
    print(f"Training complete. Outputs: {output_dir}")

