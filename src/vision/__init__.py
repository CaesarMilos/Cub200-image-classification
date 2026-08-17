"""
文件作用：标记 CUB-200-2011 视觉模块包。
File purpose: declare the core vision package without eager heavy imports.

各入口按需从具体模块导入，避免仅处理配置时也强制初始化 TorchVision 模型。
"""

__all__: list[str] = []
