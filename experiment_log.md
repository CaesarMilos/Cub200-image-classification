# CUB-200-2011 实验日志（重建训练）

> 本日志记录 2026-08-06 在当前工程环境完成的四组训练与一次官方 Test 集最终评估。
> **模型选择仅使用验证集；官方 Test 集仅在四组模型训练完成、权重冻结后统一评估一次。**

## 1. 实验协议与数据集约定

| 项目 | 记录 |
| --- | --- |
| 数据集 | 官方 CUB-200-2011，200 个鸟类类别，共 11,788 张图像 |
| 固定划分 | Train 5,400 / Validation 594 / Official Test 5,794 |
| 划分随机种子 | `42` |
| 数据准备入口 | `prepare_data.py` |
| 模型选择依据 | Validation Accuracy 与 Validation Macro-F1 |
| 冻结权重 | 每组的 `best_val_raw.pt` |
| 最终评估入口 | `evaluate_all_final.py` |
| 可视化入口 | `generate_final_visuals.py` |

每次训练运行目录均保留以下可复现证据：

- `resolved_config.yaml`：本次实际解析后的配置；
- `history.csv`：逐 Epoch 训练/验证历史；
- `best_val_raw.pt`：验证集最佳的原始权重。

最终官方 Test 评估还保留 `metrics.json`、`predictions.csv`、`per_class_metrics.csv`、`top_confusion_pairs.csv` 与混淆矩阵数组等证据文件。

---

## 2. E1：Custom CNN（从零训练基线）

| 字段 | 记录 |
| --- | --- |
| 实验名称 | `custom_cnn_v2` |
| 训练入口 | `train_custom.py` |
| 配置文件 | `configs/custom_cnn.yaml` |
| 运行目录 | `outputs/custom_cnn_v2/20260806_160044/` |
| 模型 | 自定义 CNN，从零开始训练 |
| 训练轮数 | 120 Epoch |
| 选模权重 | `best_val_raw.pt` |

### 验证集结果（用于选模）

| 指标 | 结果 |
| --- | ---: |
| 最终训练 Accuracy | 99.57% |
| 最佳验证 Accuracy | 38.89% |
| 最佳验证 Macro-F1 | 37.06% |

### 官方 Test 集结果（冻结权重后的最终评估）

| Accuracy | Macro-F1 | Top-k Accuracy |
| ---: | ---: | ---: |
| 36.64% | 36.41% | 55.51% |

### 结论

该模型几乎记忆了训练集，但在验证集和官方 Test 集上的泛化能力明显不足，存在较强过拟合。它作为**不依赖预训练的从零训练基线**保留，用于突出后续迁移学习模型带来的性能提升。EMA 权重分支表现不佳，因此不用于该模型的最终展示或评估。

---

## 3. E2：ResNet-50（ImageNet 预训练迁移学习基线）

| 字段 | 记录 |
| --- | --- |
| 实验名称 | `resnet50_v2` |
| 训练入口 | `train_resnet.py` |
| 配置文件 | `configs/resnet50.yaml` |
| 运行目录 | `outputs/resnet50_v2/20260806_165555/` |
| 模型 | ImageNet 预训练 ResNet-50，面向 CUB-200 微调 |
| 选模权重 | `best_val_raw.pt` |

### 验证集结果（用于选模）

| 指标 | 结果 |
| --- | ---: |
| 最终验证 Accuracy | 84.51% |
| 最终验证 Macro-F1 | 83.50% |

### 官方 Test 集结果（冻结权重后的最终评估）

| Accuracy | Macro-F1 | Top-k Accuracy |
| ---: | ---: | ---: |
| 84.57% | 84.49% | 93.87% |

### 结论

ResNet-50 的 Test Accuracy 比 Custom CNN 提升 **47.93 个百分点**，清楚证明 ImageNet 预训练迁移学习对 CUB-200 细粒度分类非常有效。该模型是稳定、具有代表性的强基线。

---

## 4. E3：ConvNeXt-Tiny Baseline

| 字段 | 记录 |
| --- | --- |
| 实验名称 | `convnext_baseline_v2` |
| 训练入口 | `train_eval_strong.py` |
| 配置文件 | `configs/convnext_baseline.yaml` |
| 运行目录 | `outputs/convnext_baseline_v2/20260806_171738/` |
| 模型 | ConvNeXt-Tiny 基线版（无 Attention Pooling） |
| 训练轮数 | 100 Epoch |
| 选模权重 | `best_val_raw.pt` |

### 验证集结果（用于选模）

| 指标 | 结果 |
| --- | ---: |
| 最终训练 Accuracy | 99.93% |
| 最终验证 Accuracy | 86.53% |
| 最终验证 Macro-F1 | 85.49% |
| 训练过程中最高验证 Accuracy | 87.04%（Epoch 98） |

### 官方 Test 集结果（冻结权重后的最终评估）

| Accuracy | Macro-F1 | Top-k Accuracy |
| ---: | ---: | ---: |
| 86.24% | 86.10% | 93.98% |

### 结论

ConvNeXt-Tiny 在官方 Test 集上比 ResNet-50 高 **1.67 个百分点**，说明现代卷积网络架构更适合本任务中的细粒度视觉特征提取。训练准确率接近 100%，仍有一定过拟合，但整体泛化表现稳定，是 Attention 版本的有效对照基线。

---

## 5. E4：ConvNeXt-Tiny + Attention Pooling

| 字段 | 记录 |
| --- | --- |
| 实验名称 | `convnext_attention_v2` |
| 训练入口 | `train_eval_strong.py` |
| 配置文件 | `configs/convnext_attention.yaml` |
| 运行目录 | `outputs/convnext_attention_v2/20260806_173206/` |
| 模型 | ConvNeXt-Tiny + Attention Pooling |
| 训练轮数 | 120 Epoch |
| 选模权重 | `best_val_raw.pt` |

### 验证集结果（用于选模）

| 指标 | 结果 |
| --- | ---: |
| 最终训练 Accuracy | 99.89% |
| 最终验证 Accuracy | 87.71% |
| 最终验证 Macro-F1 | 87.13% |
| 训练过程中最高验证 Accuracy | 88.05%（Epoch 119） |

### 官方 Test 集结果（冻结权重后的最终评估）

| Accuracy | Macro-F1 | Top-k Accuracy |
| ---: | ---: | ---: |
| **87.50%** | **87.40%** | **95.48%** |

### 结论

这是四组模型中表现最优的模型。相比 ConvNeXt-Tiny Baseline，其 Test Accuracy 提升 **1.26 个百分点**，Top-k Accuracy 提升 **1.50 个百分点**。该结果说明 Attention Pooling 能更有效地聚合鸟类头部、翅膀和羽毛纹理等局部细粒度特征。

补充实验中，TTA 的 Accuracy 与 Macro-F1 几乎未带来实际增益；因此最终报告以普通推理的 `final_test_raw` 结果为主，TTA 仅作为已验证的补充结果说明。

---

## 6. 官方 Test 集横向对比

| 模型 | Test Accuracy | Macro-F1 | Top-k Accuracy | 相对 Custom CNN 的 Accuracy 提升 |
| --- | ---: | ---: | ---: | ---: |
| Custom CNN | 36.64% | 36.41% | 55.51% | — |
| ResNet-50 | 84.57% | 84.49% | 93.87% | +47.93 pt |
| ConvNeXt-Tiny Baseline | 86.24% | 86.10% | 93.98% | +49.60 pt |
| **ConvNeXt-Tiny + Attention Pooling** | **87.50%** | **87.40%** | **95.48%** | **+50.86 pt** |

## 7. 最终结论

1. 最终选用模型为 **ConvNeXt-Tiny + Attention Pooling**，其官方 Test Accuracy 为 **87.50%**，Macro-F1 为 **87.40%**。
2. 从零训练的 Custom CNN 存在明显过拟合；预训练迁移学习是本任务性能跃升的关键。
3. ConvNeXt-Tiny 优于 ResNet-50，Attention Pooling 在此基础上继续带来稳定提升。
4. 最终结果来自此前未参与选模的 5,794 张官方 Test 图像，因此可作为项目报告和作品集中的最终性能结论。

## 8. 最终可视化产物（最佳模型）

以下图像由 `generate_final_visuals.py` 基于最佳 Attention 模型生成，供最终报告与作品集使用：

- 混淆矩阵：`results/figures/attention_confusion_matrix.png`
- 高频易混类别对：`results/figures/top_confusion_pairs.png`
- 预测案例图：`results/figures/prediction_gallery.png`
- Grad-CAM 注意力热力图：`results/figures/gradcam_gallery.png`
