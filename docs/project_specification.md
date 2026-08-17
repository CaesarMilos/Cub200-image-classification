# CUB-200-2011 细粒度鸟类分类项目说明书

## 项目定位

本项目由 CSC8637 Deep Learning Coursework 的 Task 1 升级而来，目标是在 CUB-200-2011 上构建一个可复现、可解释、可交互展示的细粒度图像分类工程。

它同时服务两个场景：课程作业需要标准模型、自定义模型、指标和现场演示；AI 求职作品集需要规范数据协议、模型对照、消融实验、可复现 checkpoint、错误分析与 Demo。

该项目仅面向 CUB-200-2011 定义的 200 个类别，不应被宣称为现实世界通用鸟类识别产品。

## 问题定义

输入是一张 RGB 鸟类图像；输出包括 Top-1/Top-3 类别、置信度，以及有标签批量数据上的 Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1、Top-3 Accuracy 和混淆矩阵。单图展示额外输出 Grad-CAM。

## 数据协议

原始目录必须包含：

```text
images/
images.txt
image_class_labels.txt
classes.txt
train_test_split.txt
```

官方数据共有 11,788 张图片、200 类。`train_test_split.txt` 是不可改变的边界：official train 仅拆为 train/validation；official test 仅用于最终一次性评估。

| Split | 来源 | 用途 | 可否用于模型选择 |
|---|---|---|---|
| train | official train 的 90% | 参数优化 | 可以 |
| validation | official train 的 10% | 选择 epoch、模型和超参数 | 可以 |
| test | official test | 最终报告 | 不可以 |

`prepare_data.py` 以 seed=42 按类别分层生成以下资产，不复制整套图片：

```text
train_manifest.csv
val_manifest.csv
test_manifest.csv
class_to_idx.json
split_report.json
dataset_stats.json
```

每条 manifest 记录 image ID、图片相对路径、class ID、class name 和 split。`class_to_idx.json` 由官方 `classes.txt` 的类别 ID 顺序生成，是训练、评估和 Demo 的唯一标签空间。

数据准备必须验证：三份 split 无图像重叠；test 与官方 test 完全一致；每个 split 均覆盖 200 类；图片路径和标签文件的类别一致；原图缺失时显式失败。

## 模型设计

| 模型 | 定位 | 设计目的 |
|---|---|---|
| `CustomFineGrainedCNN` | 自定义、从零训练 | 证明残差、多尺度卷积、SE 注意力和训练策略能力 |
| ResNet-50 | 标准迁移学习 baseline | 衡量 ImageNet 预训练收益 |
| ConvNeXt-Tiny | 现代 CNN baseline | 衡量 backbone 更新带来的收益 |
| ConvNeXt + Attention Pooling | 主力模型 | 聚焦头部、翼部、羽毛等局部判别区域 |

### CustomFineGrainedCNN

```text
RGB image
→ Conv stem + max pooling
→ 4 个 Multi-Scale Residual Stages
→ Global Average Pooling
→ Dropout + 200-class classifier
```

每个多尺度残差块并行使用普通 3×3 卷积和 dilation=2 的 3×3 卷积，随后经过 1×1 融合、SE 通道注意力和残差连接。它不加载 ImageNet 权重。

### Attention Pooling

ConvNeXt-Attention 对最后一层空间特征 \(x_i\) 学习位置权重：

\[
a_i = \operatorname{softmax}(w^T x_i), \qquad z = \sum_i a_i x_i
\]

其中 \(z\) 为加权全局表示。该设计的目标是提升细粒度局部证据的利用率，而不是宣称其一定优于所有全局池化方案。

## 训练与 checkpoint 协议

三个训练入口共同使用统一 `Trainer`：

```text
固定 seed
→ 读取 manifests + class map
→ 构建模型和 transform
→ train split 训练
→ validation split 评估
→ 保存最佳 raw / EMA checkpoint
→ 模型和配置冻结后才可运行 official test
```

训练策略包括 AdamW、backbone/head 分组学习率、warmup + cosine、label smoothing、AMP、可选 EMA、RandAugment、Random Erasing 和 gradient clipping。

新版 checkpoint 保存模型结构、预处理、类别映射、raw 权重、EMA 权重、优化器/调度器状态、最佳 validation 指标、epoch、seed 与运行环境。评估、课程 Demo 和 Gradio 页面均从同一 checkpoint 恢复，而不是依赖目录排序或临时 `--arch` 参数。

## 实验协议

| 实验 | Attention | EMA | TTA | 目的 |
|---|---:|---:|---:|---|
| E1 | Custom CNN | - | - | 从零训练对照 |
| E2 | ResNet-50 | - | - | 标准迁移学习对照 |
| A | 否 | 否 | 否 | ConvNeXt baseline |
| B | 是 | 否 | 否 | Attention 的独立贡献 |
| C | 是 | 是 | 否 | EMA 的独立贡献 |
| D | 是 | 是 | 是 | C 的固定 checkpoint 上的 TTA 增益 |

所有结构与超参数决策只能使用 validation。D 不重新训练；只比较 C checkpoint 的普通推理和原图/水平翻转 logits 平均后的推理。

## 输出与展示

`evaluate.py` 的正式输出：

```text
metrics.json
predictions.csv
per_class_metrics.csv
top_confusion_pairs.csv
confusion_matrix.npy
confusion_matrix.png
```

`demo_batch.py` 接受任意图片目录与 `image,class_name` 格式的标签 CSV，输出课程所需的逐图标签、Accuracy、Precision、Recall、F1 和混淆矩阵。它不会因现场仅给出少数类别而发生类别索引错位。

`app.py` 启动 Gradio 页面，提供单图 Top-3、置信度和 Grad-CAM。

最终作品集至少应展示模型对比表、最易混淆类别对、高置信度错误样例、训练曲线和多张 Grad-CAM，并解释遮挡、背景、姿态与相似羽毛纹理等失败模式。

## 已知限制

- 仅适用于 CUB 定义的 200 类；
- 对遮挡、极端姿态、分布外图像和背景偏差可能敏感；
- TTA 会增加推理成本，必须明确报告是否开启；
- Grad-CAM 是后验可视化，不是因果解释；
- Custom CNN 的价值是可解释的工程对照，不应期待一定击败大型预训练模型。

