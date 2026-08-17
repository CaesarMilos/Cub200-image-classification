"""
文件作用：解析 CUB-200-2011 官方 metadata，生成固定数据划分与 Dataset。
File purpose: parse CUB metadata, create deterministic manifests, and build datasets.

新版训练直接通过 manifest 引用原始 images/，避免复制整套图片。所有类别编号均由
classes.txt 建立并持久化，训练、测试和现场 Demo 不再依赖目录临时排序。
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import PreprocessConfig


@dataclass(frozen=True)
class ImageRecord:
    """一张图片在工程中的固定身份与 split。/ Canonical image manifest record."""

    image_id: int
    relative_path: str
    class_id: int
    class_name: str
    split: str


@dataclass
class CUBMetadata:
    """从 CUB 官方文本文件加载的完整 metadata。/ Parsed official metadata."""

    image_paths: dict[int, str]
    image_class_ids: dict[int, int]
    official_train_flags: dict[int, bool]
    class_names: dict[int, str]


class ClassMap:
    """200 个类别名称与零基索引之间的唯一映射。/ Stable label space."""

    def __init__(self, class_names_by_id: dict[int, str]) -> None:
        ordered = sorted(class_names_by_id.items())
        self.class_to_idx = {name: index for index, (_, name) in enumerate(ordered)}
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 兼容结构。"""

        return {
            "class_to_idx": self.class_to_idx,
            "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "ClassMap":
        """从持久化 JSON 恢复类别映射。"""

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        instance = cls.__new__(cls)
        instance.class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
        instance.idx_to_class = {int(k): str(v) for k, v in payload["idx_to_class"].items()}
        return instance

    def save(self, path: str | Path) -> None:
        """保存类别映射。"""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)


class ManifestImageDataset(Dataset):
    """从 CSV manifest 读取图像的训练/验证/测试 Dataset。"""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        raw_dir: str | Path,
        class_map: ClassMap,
        transform=None,
    ) -> None:
        self.records = list(records)
        self.images_dir = Path(raw_dir) / "images"
        self.class_map = class_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.images_dir / record.relative_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = self.class_map.class_to_idx[record.class_name]
        return image, target, str(image_path)


class LabelledImageDataset(Dataset):
    """读取老师现场提供的 images + labels.csv，不要求包含全部 200 类目录。"""

    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        class_map: ClassMap,
        transform=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.class_map = class_map
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        with Path(labels_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("image") or row.get("filename")
                class_name = row.get("class_name") or row.get("label")
                if not filename or not class_name:
                    raise ValueError("labels.csv requires image/filename and class_name/label columns")
                if class_name not in class_map.class_to_idx:
                    raise ValueError(f"Unknown class name in labels.csv: {class_name}")
                self.samples.append((self.images_dir / filename, class_map.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target, str(image_path)


def _read_mapping(path: Path, value_cast=str) -> dict[int, object]:
    """读取 `<integer id> <value>` 格式的 CUB metadata 文件。"""

    result: dict[int, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            key, value = stripped.split(maxsplit=1)
            numeric_key = int(key)
            if numeric_key in result:
                raise ValueError(f"Duplicate id {numeric_key} in {path} line {line_number}")
            result[numeric_key] = value_cast(value)
    return result


def load_cub_metadata(raw_dir: str | Path) -> CUBMetadata:
    """读取并交叉校验 CUB 官方 metadata。/ Load and cross-check metadata."""

    root = Path(raw_dir)
    metadata = CUBMetadata(
        image_paths=_read_mapping(root / "images.txt", str),
        image_class_ids=_read_mapping(root / "image_class_labels.txt", int),
        official_train_flags=_read_mapping(root / "train_test_split.txt", lambda v: bool(int(v))),
        class_names=_read_mapping(root / "classes.txt", str),
    )
    image_ids = set(metadata.image_paths)
    if image_ids != set(metadata.image_class_ids) or image_ids != set(metadata.official_train_flags):
        raise ValueError("images.txt, labels, and official split do not contain identical image IDs")
    unknown_classes = set(metadata.image_class_ids.values()).difference(metadata.class_names)
    if unknown_classes:
        raise ValueError(f"Unknown class IDs referenced by images: {sorted(unknown_classes)}")
    mismatched_paths = []
    for image_id, relative_path in metadata.image_paths.items():
        expected_class = metadata.class_names[metadata.image_class_ids[image_id]]
        if Path(relative_path).parts[0] != expected_class:
            mismatched_paths.append((image_id, relative_path, expected_class))
    if mismatched_paths:
        raise ValueError(
            "Image path class folders disagree with image_class_labels.txt. "
            f"Examples: {mismatched_paths[:3]}"
        )
    return metadata


def create_stratified_records(
    metadata: CUBMetadata, val_ratio: float = 0.1, seed: int = 42
) -> list[ImageRecord]:
    """将官方 train 按类拆为 train/val，并原样保留官方 test。"""

    train_by_class: dict[int, list[int]] = {class_id: [] for class_id in metadata.class_names}
    test_ids: list[int] = []
    for image_id in sorted(metadata.image_paths):
        if metadata.official_train_flags[image_id]:
            train_by_class[metadata.image_class_ids[image_id]].append(image_id)
        else:
            test_ids.append(image_id)

    split_by_id: dict[int, str] = {}
    for class_id, image_ids in sorted(train_by_class.items()):
        if not image_ids:
            raise ValueError(f"Class {class_id} has no official training images")
        shuffled = list(image_ids)
        random.Random(seed + class_id).shuffle(shuffled)
        val_count = max(1, math.floor(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        for image_id in shuffled[:val_count]:
            split_by_id[image_id] = "val"
        for image_id in shuffled[val_count:]:
            split_by_id[image_id] = "train"
    for image_id in test_ids:
        split_by_id[image_id] = "test"

    return [
        ImageRecord(
            image_id=image_id,
            relative_path=metadata.image_paths[image_id],
            class_id=metadata.image_class_ids[image_id],
            class_name=metadata.class_names[metadata.image_class_ids[image_id]],
            split=split_by_id[image_id],
        )
        for image_id in sorted(metadata.image_paths)
    ]


def validate_split_integrity(records: Sequence[ImageRecord], metadata: CUBMetadata) -> dict[str, object]:
    """验证无重复、无泄漏、官方 test 未被改变，并返回统计信息。"""

    if len({record.image_id for record in records}) != len(records):
        raise ValueError("Duplicate image IDs detected in generated records")
    if {record.image_id for record in records} != set(metadata.image_paths):
        raise ValueError("Generated records do not cover all official images")

    split_ids = {name: {r.image_id for r in records if r.split == name} for name in ("train", "val", "test")}
    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise ValueError("Data leakage: split image IDs overlap")
    official_test = {image_id for image_id, is_train in metadata.official_train_flags.items() if not is_train}
    if split_ids["test"] != official_test:
        raise ValueError("Generated test split differs from the official CUB test split")

    per_split_classes = {
        name: len({r.class_id for r in records if r.split == name}) for name in split_ids
    }
    expected_classes = len(metadata.class_names)
    incomplete = {
        name: count for name, count in per_split_classes.items() if count != expected_classes
    }
    if incomplete:
        raise ValueError(f"Not every split contains all {expected_classes} classes: {incomplete}")
    per_class_counts = {
        split_name: {
            metadata.class_names[class_id]: sum(
                record.split == split_name and record.class_id == class_id for record in records
            )
            for class_id in sorted(metadata.class_names)
        }
        for split_name in ("train", "val", "test")
    }
    return {
        "total_images": len(records),
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "classes_per_split": per_split_classes,
        "num_classes": len(metadata.class_names),
        "per_class_counts": per_class_counts,
    }


def validate_image_files(records: Iterable[ImageRecord], raw_dir: str | Path) -> None:
    """检查所有 manifest 指向的原图是否存在；缺失时直接失败。"""

    images_dir = Path(raw_dir) / "images"
    missing = [record.relative_path for record in records if not (images_dir / record.relative_path).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. Examples: {preview}")


def write_manifests(
    records: Sequence[ImageRecord], artifacts_dir: str | Path, stats: dict[str, object]
) -> None:
    """写出三个 manifest、类别统计与 split 报告。"""

    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(ImageRecord.__annotations__)
    for split_name in ("train", "val", "test"):
        with (output / f"{split_name}_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records if record.split == split_name)
    with (output / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    with (output / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_images": stats["total_images"],
                "num_classes": stats["num_classes"],
                "split_counts": stats["split_counts"],
                "per_class_counts": stats["per_class_counts"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """从 CSV 恢复 ImageRecord 列表。"""

    records: list[ImageRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                ImageRecord(
                    image_id=int(row["image_id"]),
                    relative_path=row["relative_path"],
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    split=row["split"],
                )
            )
    return records


def build_transform(config: PreprocessConfig, training: bool):
    """根据 checkpoint/config 构建完全一致的训练或推理 transform。"""

    operations: list[object] = []
    if training and config.random_resized_crop:
        operations.append(
            transforms.RandomResizedCrop(
                config.image_size, scale=tuple(config.crop_scale), ratio=(0.75, 1.33)
            )
        )
    else:
        operations.extend(
            [transforms.Resize(config.resize_size), transforms.CenterCrop(config.image_size)]
        )
    if training:
        if config.randaugment_ops > 0:
            operations.append(
                transforms.RandAugment(
                    num_ops=config.randaugment_ops, magnitude=config.randaugment_magnitude
                )
            )
        if config.horizontal_flip > 0:
            operations.append(transforms.RandomHorizontalFlip(config.horizontal_flip))
        if config.random_rotation > 0:
            operations.append(transforms.RandomRotation(config.random_rotation))
        if config.color_jitter > 0:
            operations.append(
                transforms.ColorJitter(
                    brightness=config.color_jitter, contrast=config.color_jitter
                )
            )
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)]
    )
    if training and config.random_erasing > 0:
        operations.append(
            transforms.RandomErasing(
                p=config.random_erasing,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    return transforms.Compose(operations)
