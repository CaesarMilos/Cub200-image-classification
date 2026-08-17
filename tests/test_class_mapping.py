"""
文件作用：验证类别映射不依赖字典插入顺序，且 JSON 往返一致。
File purpose: test stable class mapping persistence.
"""

import pytest

pytest.importorskip("torch")

from src.vision.data import ClassMap


def test_class_map_is_sorted_by_official_class_id(tmp_path) -> None:
    """官方 class ID 顺序必须映射到连续零基索引。"""

    class_map = ClassMap({2: "002.Bird_B", 1: "001.Bird_A"})
    assert class_map.class_to_idx == {"001.Bird_A": 0, "002.Bird_B": 1}
    path = tmp_path / "class_to_idx.json"
    class_map.save(path)
    restored = ClassMap.from_json(path)
    assert restored.class_to_idx == class_map.class_to_idx
    assert restored.idx_to_class == class_map.idx_to_class

