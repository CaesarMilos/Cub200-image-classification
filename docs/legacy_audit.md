# 旧版源码审计摘要

本文记录改写动机，避免后续误把旧 checkpoint 当作新版正式结果。

## 已保留的有效思路

- official train/test 分离，官方训练集按类 9:1 划分 train/validation；
- ResNet-50 ImageNet 迁移学习与 448 输入；
- ConvNeXt-Tiny + Attention Pooling；
- RandAugment、Random Erasing、label smoothing、AdamW、warmup-cosine；
- EMA 与水平翻转 TTA；
- Accuracy、Macro Precision/Recall、混淆矩阵与现场逐图预测。

## 旧版必须修复的问题

- `train_eval_strong.py --model resnet50` 实际仍构建 ConvNeXt；
- EMA validation 后保存的是 raw 权重，最终 test 又使用最后 epoch EMA shadow；
- 提交 checkpoint 不含 EMA、配置、类别映射或 transform；
- `test.py` 由现场子目录重建类别索引，部分类别测试集会导致分类头或标签错位；
- ResNet/ConvNeXt 评估逻辑重复，输出会覆盖；
- 数据缺失时静默跳过，`processed_data` 会被直接删除重建；
- 当前上传源码中缺少 `train_custom.py`。

因此旧版 85.36% / 87.11% 只作为历史参考，新版需重新训练并用统一评估入口生成最终结果。

