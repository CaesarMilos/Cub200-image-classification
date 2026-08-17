"""
文件作用：保留旧项目 `test.py` 名称的兼容入口。
File purpose: backward-friendly alias for the new labelled batch demo.

新版需要显式提供 labels.csv，以避免旧版 ImageFolder 子集造成类别索引错位。
"""

from demo_batch import main


if __name__ == "__main__":
    main()

