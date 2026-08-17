"""
文件作用：生成 CUB train/val/test manifests、固定类别映射并校验数据完整性。
File purpose: create deterministic split artifacts without copying the image dataset.

关键参数位于 configs/data.yaml；`verify_images` 建议正式运行时保持 true。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.vision.data import (
    ClassMap,
    create_stratified_records,
    load_cub_metadata,
    validate_image_files,
    validate_split_integrity,
    write_manifests,
)


def main() -> None:
    """执行 metadata 读取、分层划分、校验和落盘。"""

    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 manifest files")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metadata = load_cub_metadata(config["raw_dir"])
    records = create_stratified_records(
        metadata,
        val_ratio=float(config.get("val_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
    )
    stats = validate_split_integrity(records, metadata)
    if bool(config.get("verify_images", True)):
        validate_image_files(records, config["raw_dir"])

    artifacts_dir = Path(config["artifacts_dir"])
    write_manifests(records, artifacts_dir, stats)
    ClassMap(metadata.class_names).save(artifacts_dir / "class_to_idx.json")
    print("Data preparation complete")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Split counts: {stats['split_counts']}")


if __name__ == "__main__":
    main()

