"""
文件作用：集中定义 Custom CNN、ResNet-50 与 ConvNeXt-Attention 模型。
File purpose: define every model architecture behind a single model factory.

训练、评估和 Demo 必须通过 build_model() 构建模型，避免旧版在 test.py 中重复定义
Attention Head 后出现结构漂移。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import ModelConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力，强调判别性羽毛与部位特征。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class MultiScaleResidualBlock(nn.Module):
    """自定义多尺度残差块：并行 3x3 与扩张卷积分支提取局部/较大纹理。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        branch_channels = out_channels // 2
        self.branch_local = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.branch_context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.branch_local(x), self.branch_context(x)], dim=1)
        return self.activation(self.fuse(features) + self.shortcut(x))


class CustomFineGrainedCNN(nn.Module):
    """
    从零训练的自定义细粒度分类网络。

    通过四个层级的多尺度残差块逐步提取边缘、羽毛纹理、局部部位与整体形状，
    最后使用全局平均池化完成 200 类分类。该模型不加载 ImageNet 权重。
    """

    def __init__(
        self, num_classes: int, channels: list[int] | None = None, dropout: float = 0.3
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        if len(channels) != 4 or any(channel % 2 for channel in channels):
            raise ValueError("custom_channels must contain four even channel values")

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(channels[0], channels[0], blocks=2, stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], blocks=2, stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], blocks=3, stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[3], num_classes)
        )
        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [MultiScaleResidualBlock(in_channels, out_channels, stride)]
        layers.extend(MultiScaleResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """为从零训练的卷积和线性层使用 Kaiming/截断正态初始化。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回最终二维特征图，供分类与 Grad-CAM 共用。"""

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.forward_features(x)))


class AttentionPoolingHead(nn.Module):
    """学习空间权重并对 ConvNeXt 特征图加权池化的自定义分类头。"""

    def __init__(
        self, in_channels: int, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        spatial_weights = torch.softmax(self.attention(x).reshape(batch, -1), dim=1)
        flattened = x.reshape(batch, channels, -1)
        pooled = torch.sum(flattened * spatial_weights.unsqueeze(1), dim=2)
        return self.classifier(self.norm(pooled))


class ConvNeXtClassifier(nn.Module):
    """封装 ConvNeXt-Tiny，可选择标准全局池化或 Attention Pooling。"""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        attention_pooling: bool = False,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = models.convnext_tiny(weights=weights)
        self.features = base.features
        self.attention_pooling = attention_pooling
        if attention_pooling:
            self.head = AttentionPoolingHead(768, num_classes, hidden_dim, dropout)
            self.avgpool = None
            self.classifier = None
        else:
            self.avgpool = base.avgpool
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.classifier = base.classifier
            self.head = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.attention_pooling:
            return self.head(features)
        return self.classifier(self.avgpool(features))


def build_model(config: ModelConfig) -> nn.Module:
    """根据配置构建唯一且可复现的模型结构。/ Central model factory."""

    name = config.name.lower()
    if name == "custom_cnn":
        return CustomFineGrainedCNN(
            num_classes=config.num_classes,
            channels=config.custom_channels,
            dropout=config.dropout,
        )
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if config.pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(in_features, config.num_classes))
        return model
    if name in {"convnext_tiny", "convnext_attention"}:
        use_attention = config.attention_pooling or name == "convnext_attention"
        return ConvNeXtClassifier(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            attention_pooling=use_attention,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model name: {config.name}")


def split_backbone_and_head_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """将 backbone 与新分类头分组，以便使用不同学习率。"""

    head_tokens = ("fc", "head", "classifier")
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = head_parameters if any(token in name for token in head_tokens) else backbone_parameters
        target.append(parameter)
    return backbone_parameters, head_parameters


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """冻结或解冻 backbone；分类头始终保持可训练。"""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or any(
            token in name for token in ("fc", "head", "classifier")
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """统计总参数量与可训练参数量。"""

    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """返回各模型最后一个适合 Grad-CAM 的卷积/特征层。"""

    if isinstance(model, ConvNeXtClassifier):
        return model.features[-1]
    if isinstance(model, CustomFineGrainedCNN):
        return model.stage4[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise TypeError(f"No Grad-CAM target layer registered for {type(model).__name__}")

