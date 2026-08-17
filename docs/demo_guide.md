# 30 分钟课程 Demo 指南

## 演示前检查

```bash
python -c "import torch; print(torch.cuda.is_available())"
python evaluate.py --checkpoint <checkpoint> --split val --output_dir demo_precheck
```

确认 checkpoint、类别映射、输入尺寸和 transform 均可恢复。

## 老师提供图片后

1. 将图片放进一个普通目录，不需要建立 200 个类别子目录；
2. 建立 `labels.csv`，包含 `image,class_name`；
3. 运行 `demo_batch.py`；
4. 展示终端指标、`predictions.csv` 与 `confusion_matrix.png`；
5. 随机解释几张正确与错误预测。

## 必须展示的内容

- 每张图的真实标签和预测标签；
- Accuracy、Precision、Recall、F1；
- 混淆矩阵；
- 模型 checkpoint 和使用的预处理；
- 如果启用 TTA，明确说明原图与水平翻转 logits 做平均。

