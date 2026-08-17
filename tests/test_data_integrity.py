"""
文件作用：验证分层划分覆盖全部图像、保持官方 test 且不存在 split 泄漏。
File purpose: unit-test the deterministic CUB split contract.
"""

import pytest

pytest.importorskip("torch")

from src.vision.data import CUBMetadata, create_stratified_records, validate_split_integrity


def test_stratified_records_keep_official_test() -> None:
    """两类合成 metadata 应得到互斥 train/val/test。"""

    metadata = CUBMetadata(
        image_paths={i: f"class_{1 if i <= 6 else 2}/image_{i}.jpg" for i in range(1, 13)},
        image_class_ids={i: 1 if i <= 6 else 2 for i in range(1, 13)},
        official_train_flags={i: i not in {6, 12} for i in range(1, 13)},
        class_names={1: "001.Class_One", 2: "002.Class_Two"},
    )
    records = create_stratified_records(metadata, val_ratio=0.2, seed=42)
    stats = validate_split_integrity(records, metadata)
    assert stats["total_images"] == 12
    assert stats["split_counts"]["test"] == 2
    assert {record.image_id for record in records if record.split == "test"} == {6, 12}

