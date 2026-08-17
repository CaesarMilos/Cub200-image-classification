"""
文件作用：统一模型训练、EMA、warmup-cosine 调度与最佳 checkpoint 选择。
File purpose: provide the shared training engine for all Task 1 models.
"""

from __future__ import annotations

import copy
import csv
import math
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, save_resolved_config
from .data import ClassMap, ManifestImageDataset, build_transform, load_manifest
from .evaluator import evaluate_loader
from .models import (
    build_model,
    set_backbone_trainable,
    split_backbone_and_head_parameters,
)
from .seed import build_dataloader_generator, collect_environment_info, seed_worker, set_global_seed


class EMA:
    """模型参数的指数滑动平均；shadow 权重可以独立保存和恢复。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """每个 optimizer step 后更新 EMA 参数。"""

        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def model_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """生成可直接 load_state_dict 的完整 EMA 模型状态。"""

        state = copy.deepcopy(model.state_dict())
        for name, value in self.shadow.items():
            if name in state:
                state[name] = value.detach().clone()
        return state

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        """在上下文中临时应用 EMA 权重，退出后恢复 raw 权重。"""

        backup = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }
        try:
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.data.copy_(self.shadow[name])
            yield
        finally:
            for name, parameter in model.named_parameters():
                if name in backup:
                    parameter.data.copy_(backup[name])


class WarmupCosineScheduler:
    """按 iteration 执行线性 warmup 后 cosine 衰减的学习率调度器。"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.02,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_number = 0

    def step(self) -> None:
        """推进一次调度；应在每个训练 batch 的 optimizer.step() 后调用。"""

        self.step_number += 1
        if self.step_number <= self.warmup_steps:
            scale = self.step_number / self.warmup_steps
        else:
            progress = (self.step_number - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[index] * scale

    def state_dict(self) -> dict[str, object]:
        """返回可写入 resume checkpoint 的调度状态。"""

        return {"step_number": self.step_number, "base_lrs": self.base_lrs}


class Trainer:
    """封装 train/validation 循环，并分别选择最佳 raw 与 EMA checkpoint。"""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        class_map: ClassMap,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.class_map = class_map
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.optimizer = self._build_optimizer()
        total_steps = len(train_loader) * config.training.epochs
        warmup_steps = len(train_loader) * config.training.warmup_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps, config.training.min_lr_ratio
        )
        self.ema = EMA(model, config.training.ema_decay) if config.training.ema else None
        self.best_scores = {"raw": float("-inf"), "ema": float("-inf")}
        self.environment = collect_environment_info()

        use_amp = config.training.amp and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

    def _build_optimizer(self) -> Optimizer:
        """为 backbone 与分类头建立分组 AdamW。"""

        backbone, head = split_backbone_and_head_parameters(self.model)
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": self.config.training.backbone_lr})
        if head:
            groups.append({"params": head, "lr": self.config.training.head_lr})
        return AdamW(groups, weight_decay=self.config.training.weight_decay)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """训练一个 epoch 并返回 loss/accuracy。"""

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, targets, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item() * images.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_seen += images.size(0)
        return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen}

    def _validation_metrics(self) -> dict[str, float | int]:
        """调用统一 evaluator 计算 validation 指标。"""

        result = evaluate_loader(
            self.model,
            self.val_loader,
            self.device,
            self.class_map,
            top_k=self.config.evaluation.top_k,
            tta_horizontal_flip=False,
        )
        return result.metrics.to_dict()

    def _maybe_save_best(
        self,
        variant: str,
        epoch: int,
        metrics: dict[str, float | int],
    ) -> bool:
        """根据 validation 指标决定是否保存 raw/EMA 最佳 checkpoint。"""

        metric_name = self.config.training.selection_metric
        value = float(metrics[metric_name])
        if value <= self.best_scores[variant]:
            return False
        self.best_scores[variant] = value
        ema_state = self.ema.model_state_dict(self.model) if self.ema is not None else None
        save_checkpoint(
            self.output_dir / f"best_val_{variant}.pt",
            config=self.config,
            class_map=self.class_map,
            raw_state_dict=copy.deepcopy(self.model.state_dict()),
            ema_state_dict=ema_state,
            selected_weights=variant,
            epoch=epoch,
            validation_metrics=metrics,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            environment=self.environment,
        )
        return True

    def fit(self) -> list[dict[str, float | int | str]]:
        """执行完整训练并保存 history.csv、resolved config 与最佳 checkpoint。"""

        save_resolved_config(self.config, self.output_dir / "resolved_config.yaml")
        history: list[dict[str, float | int | str]] = []
        freeze_epochs = self.config.training.freeze_backbone_epochs
        if freeze_epochs > 0:
            set_backbone_trainable(self.model, False)

        for epoch in range(1, self.config.training.epochs + 1):
            if epoch == freeze_epochs + 1 and freeze_epochs > 0:
                set_backbone_trainable(self.model, True)
            start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            raw_metrics = self._validation_metrics()
            raw_saved = self._maybe_save_best("raw", epoch, raw_metrics)
            ema_metrics = None
            ema_saved = False
            if self.ema is not None:
                with self.ema.apply(self.model):
                    ema_metrics = self._validation_metrics()
                ema_saved = self._maybe_save_best("ema", epoch, ema_metrics)

            row: dict[str, float | int | str] = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy_raw": raw_metrics["accuracy"],
                "val_macro_f1_raw": raw_metrics["macro_f1"],
                "val_accuracy_ema": ema_metrics["accuracy"] if ema_metrics else "",
                "val_macro_f1_ema": ema_metrics["macro_f1"] if ema_metrics else "",
                "best_raw_saved": str(raw_saved),
                "best_ema_saved": str(ema_saved),
                "seconds": time.time() - start,
            }
            history.append(row)
            print(
                f"Epoch {epoch:03d}/{self.config.training.epochs} | "
                f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
                f"val raw acc={raw_metrics['accuracy']:.4f} f1={raw_metrics['macro_f1']:.4f}"
                + (
                    f" | val EMA acc={ema_metrics['accuracy']:.4f} f1={ema_metrics['macro_f1']:.4f}"
                    if ema_metrics
                    else ""
                )
            )

        with (self.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        return history


def build_training_dataloaders(
    config: ExperimentConfig, class_map: ClassMap
) -> tuple[DataLoader, DataLoader]:
    """从固定 manifest 构建带 worker seed 的 train/val DataLoader。"""

    artifacts = Path(config.data.artifacts_dir)
    train_dataset = ManifestImageDataset(
        load_manifest(artifacts / "train_manifest.csv"),
        config.data.raw_dir,
        class_map,
        build_transform(config.preprocess, training=True),
    )
    val_dataset = ManifestImageDataset(
        load_manifest(artifacts / "val_manifest.csv"),
        config.data.raw_dir,
        class_map,
        build_transform(config.preprocess, training=False),
    )
    common = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": build_dataloader_generator(config.training.seed),
        "persistent_workers": config.data.num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=False, **common),
        DataLoader(val_dataset, shuffle=False, drop_last=False, **common),
    )


def run_training_experiment(config: ExperimentConfig) -> Path:
    """训练入口共用的高层流程：seed → 数据 → 模型 → Trainer。"""

    set_global_seed(config.training.seed, config.training.deterministic)
    artifacts = Path(config.data.artifacts_dir)
    class_map = ClassMap.from_json(artifacts / "class_to_idx.json")
    if len(class_map.class_to_idx) != config.model.num_classes:
        raise ValueError("model.num_classes differs from persisted class mapping")
    train_loader, val_loader = build_training_dataloaders(config, class_map)
    model = build_model(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 每次训练创建独立 run 目录，防止覆盖旧 checkpoint 与 history。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.root_dir) / config.output.experiment_name / run_id
    trainer = Trainer(model, config, class_map, train_loader, val_loader, device, output_dir)
    trainer.fit()
    return output_dir
