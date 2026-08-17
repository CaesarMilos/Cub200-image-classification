# CUB-200 Fine-Grained Bird Classification

本项目将 CSC8637 Task 1 课程作业升级为可复现、可解释、可交互展示的细粒度图像分类工程。项目比较从零训练的 Custom CNN、ImageNet 迁移学习 ResNet-50，以及 ConvNeXt-Tiny + 自定义 Attention Pooling。

## 核心目标

- 严格遵循 CUB-200-2011 官方 train/test 划分；official test 不参与训练或调参。
- 官方训练集按类别、seed=42 固定拆成 90% train 和 10% validation。
- 所有模型共享同一类别映射、manifest、训练引擎、checkpoint 和评估实现。
- checkpoint 自带模型结构、预处理、类别映射、raw/EMA 权重和验证指标。
- 支持课程批量 Demo，以及求职作品集的 Top-3 + Grad-CAM 页面。

历史课程版本曾报告 ResNet-50 85.36% 与 ConvNeXt-Attention 87.11% test Accuracy。由于旧版 EMA/TTA checkpoint 链路不能严格独立复现，这些数字在新版中标记为 `legacy historical results`；新版正式结果应在重新训练并执行统一 `evaluate.py` 后填写。

## 工程结构

```text
configs/                 YAML 实验配置
src/vision/data.py       metadata、manifest、Dataset、transform
src/vision/models.py     Custom CNN、ResNet、ConvNeXt、Attention Head
src/vision/engine.py     AMP、EMA、warmup-cosine、Trainer
src/vision/checkpoint.py 自描述 checkpoint
src/vision/evaluator.py  统一批量推理与结果导出
src/vision/visualization.py 混淆矩阵与 Grad-CAM
prepare_data.py          数据准备入口
train_*.py               三类模型训练入口
evaluate.py              固定 val/test 正式评估
demo_batch.py            老师现场有标签图片集评估
app.py                   Gradio 单图演示
migrate_legacy_checkpoint.py 旧 state_dict 兼容迁移（仅 raw 权重）
```

## 推荐目录

```text
E:/AI_Projects/CUB_200_Classification/    本代码仓库
E:/Datasets/CUB_200_2011/                CUB 原始数据与 manifests
E:/Models/CUB_200_Classification/        checkpoint 与训练日志
```

原始数据目录至少需要：

```text
images/
images.txt
image_class_labels.txt
classes.txt
train_test_split.txt
```

## VS Code 一键运行

本工程已包含 `.vscode/launch.json`。在 VS Code 中选择 `py311` Python 解释器后，点击左侧“运行和调试”，从下拉列表选择对应任务，再点击绿色运行按钮或按 `F5` 即可：

- `1. Prepare CUB data`：生成固定数据资产；
- `2` 至 `5`：分别训练四组模型；
- `6. Evaluate final test`：弹窗填写训练产生的 checkpoint 路径，自动执行正式测试；
- `7. Launch portfolio demo`：弹窗填写 checkpoint 路径，启动 Gradio 展示页。

`prepare_data.py`、三个训练文件也可以直接打开后点击右上角 Python 运行按钮；不过建议统一使用上述“运行和调试”配置，参数更明确，也方便设置断点。

## 环境安装

推荐 Python 3.11。RTX 5090 用户应先按照本机驱动与 CUDA 兼容情况安装官方 PyTorch GPU 版本，再安装其余依赖：

```bash
conda create -n cub200-portfolio python=3.11 -y
conda activate cub200-portfolio
pip install -r requirements.txt
```

随后验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. 生成固定数据资产

先修改 `configs/data.yaml` 中的 E 盘路径：

```bash
python prepare_data.py --config configs/data.yaml
```

输出：

```text
artifacts/
├── train_manifest.csv
├── val_manifest.csv
├── test_manifest.csv
├── class_to_idx.json
├── split_report.json
└── dataset_stats.json
```

该步骤会检查 split 是否重叠、official test 是否被改动、metadata 是否一致以及所有图片是否存在。它不会复制整套图片，也不会删除原始数据。

## 2. 训练三个模型

```bash
python train_custom.py --config configs/custom_cnn.yaml
python train_resnet.py --config configs/resnet50.yaml
python train_eval_strong.py --config configs/convnext_attention.yaml
```

可用 `--set` 临时覆盖配置，适合 smoke test：

```bash
python train_eval_strong.py --set training.epochs=2 --set data.batch_size=8
```

每个实验目录包含：

```text
<experiment_name>/<YYYYMMDD_HHMMSS>/
├── best_val_raw.pt
├── best_val_ema.pt          # 配置启用 EMA 时生成
├── history.csv
└── resolved_config.yaml
```

训练入口只使用 train/validation，不会自动运行 official test。

## 3. 最终评估

模型与超参数完全冻结后，再执行 official test：

```bash
python evaluate.py \
  --checkpoint E:/Models/CUB_200_Classification/convnext_attention_v2/<run_id>/best_val_ema.pt \
  --split test \
  --tta \
  --output_dir E:/Models/CUB_200_Classification/convnext_attention_v2/<run_id>/final_test
```

输出包括 Accuracy、Macro Precision/Recall/F1、Weighted F1、Top-3 Accuracy、每类指标、逐图预测、混淆矩阵和 Top confusion pairs。

## 可选：迁移旧版 raw 权重

旧 checkpoint 没有 EMA、类别映射和配置，不能直接作为新版最终结果，但可以包装后进行历史对照：

```bash
python migrate_legacy_checkpoint.py \
  --legacy_checkpoint path/to/best_convnext_attn.pth \
  --config configs/convnext_attention.yaml \
  --output E:/Models/CUB_200_Classification/legacy_convnext_raw_v2.pt \
  --legacy_test_accuracy 0.8711
```

迁移只能恢复 raw state_dict；不能重建旧训练过程中未保存的 EMA shadow。

## 4. 课程现场 Demo

老师提供的 `labels.csv` 使用以下格式：

```csv
image,class_name
bird_001.jpg,001.Black_footed_Albatross
bird_002.jpg,002.Laysan_Albatross
```

运行：

```bash
python demo_batch.py \
  --checkpoint E:/Models/CUB_200_Classification/convnext_attention_v2/<run_id>/best_val_ema.pt \
  --images_dir path/to/demo_images \
  --labels_csv path/to/labels.csv \
  --output_dir demo_results \
  --tta
```

即使现场图片只包含少数类别，模型仍使用 checkpoint 保存的完整 200 类映射，不会出现旧版 `ImageFolder` 索引错位。

## 5. 作品集交互 Demo

```bash
python app.py --checkpoint E:/Models/CUB_200_Classification/convnext_attention_v2/<run_id>/best_val_ema.pt
```

页面显示 Top-3 类别、置信度和 Grad-CAM 模型关注区域。

## 消融协议

最低四组实验：

| 实验 | Attention | EMA | TTA | 目的 |
|---|---:|---:|---:|---|
| A | 否 | 否 | 否 | ConvNeXt baseline |
| B | 是 | 否 | 否 | Attention Pooling 贡献 |
| C | 是 | 是 | 否 | EMA 贡献 |
| D | 是 | 是 | 是 | 同一 C checkpoint 的推理增强 |

所有结构与训练决策只根据 validation；D 不重新训练，只在固定 checkpoint 上切换 TTA。

## 测试

```bash
pytest -q
python -m compileall -q .
```

更完整的项目边界、架构和工作流见 `docs/project_specification.md`；逐项本机验收标准见 `docs/acceptance_checklist.md`。实验规则见 `docs/experiment_protocol.md`，现场步骤见 `docs/demo_guide.md`。
