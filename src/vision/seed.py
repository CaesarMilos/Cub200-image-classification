"""
文件作用：集中控制可复现性并记录运行环境。
File purpose: configure reproducibility and collect runtime environment metadata.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """固定 Python、NumPy、PyTorch 与 CUDA 随机数生成器。"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def seed_worker(worker_id: int) -> None:
    """为每个 DataLoader worker 设置稳定且互不相同的随机种子。"""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader_generator(seed: int) -> torch.Generator:
    """创建由固定 seed 控制的 DataLoader shuffle 生成器。"""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def collect_environment_info() -> dict[str, Any]:
    """收集写入 checkpoint 的环境信息，便于跨机器复现。"""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # Convert TorchVersion to str so PyTorch 2.6 weights_only loading stays portable.
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
