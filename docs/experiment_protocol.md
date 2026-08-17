# 实验协议

## 数据边界

1. official train 仅用于 train/validation；official test 永远不进入训练。
2. train/validation 由 `prepare_data.py` 使用 seed=42 固定生成。
3. manifest 生成后不得为了获得更高 validation/test 结果更换 seed。
4. 所有模型必须使用同一批 manifest 与 `class_to_idx.json`。

## 模型选择

- epoch checkpoint 只能根据 validation Accuracy 或 Macro F1 选择；
- raw 与 EMA 分别维护最佳 validation checkpoint；
- test 指标不能反向决定训练超参数；
- TTA 是推理策略，必须和非 TTA 结果分开报告。

## 最小实验矩阵

1. Custom CNN：从零训练；
2. ResNet-50：标准迁移学习；
3. ConvNeXt-Tiny baseline；
4. ConvNeXt + Attention；
5. ConvNeXt + Attention + EMA；
6. 固定 EMA checkpoint + TTA。

## 每个实验必须保存

- `resolved_config.yaml`；
- `history.csv`；
- 最佳 checkpoint；
- validation 指标；
- 最终选中模型的 test `metrics.json`、`predictions.csv` 和混淆矩阵；
- Python/PyTorch/CUDA/GPU 环境信息。

