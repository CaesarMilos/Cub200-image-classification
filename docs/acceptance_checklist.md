# 完整验收说明与发布清单

本清单定义“项目可以作为课程交付和 AI 求职作品集发布”的最低标准。所有 `[ ]` 项应在你的 `py311` + RTX 5090 环境中完成，并保留对应输出文件作为证据。

## A. 源码与文档

- [x] 根目录保留 `prepare_data.py`、`train_custom.py`、`train_resnet.py`、`train_eval_strong.py`、`test.py` 等课程可识别入口；
- [x] 公共实现集中在 `src/vision/`，训练/评估不再重复实现；
- [x] 每个 Python 文件有文件作用说明；
- [x] 每个类有功能说明；
- [x] 每个 YAML 的关键参数有注释；
- [x] `.gitignore` 排除数据、checkpoint、缓存和本地输出；
- [x] 项目 README、项目说明、实验协议、模型卡和 Demo 指南存在。

验收命令：

```bash
python -m compileall -q .
pytest -q
```

通过标准：无 syntax error；pytest 全部通过。任何 skip 都必须说明原因，不能把 skip 当作完整通过。

## B. 环境

- [ ] 在 `py311` 环境成功执行 `pip install -r requirements.txt`；
- [ ] `torch.cuda.is_available()` 为 `True`；
- [ ] 终端显示 RTX 5090；
- [ ] TorchVision 预训练权重可下载或已缓存；
- [ ] `gradio` 可正常 import；
- [ ] `pytest -q` 不再因缺少 PyTorch 而跳过模型测试。

```bash
python -c "import torch, torchvision, gradio; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## C. 数据

- [ ] `configs/data.yaml` 指向正确的 CUB 原始目录；
- [ ] `python prepare_data.py --config configs/data.yaml` 成功；
- [ ] `class_to_idx.json` 恰有 200 类；
- [ ] `split_report.json` 显示 train=5400、val=594、test=5794；
- [ ] 所有 split 均有 200 类；
- [ ] 项目没有复制 `images/`，也没有删除原始数据；
- [ ] 故意缺失一张图片时流程会显式失败，而不是静默跳过。

通过标准：test manifest 与官方 `train_test_split.txt` 完全对应；不能手动编辑 manifest 以优化结果。

## D. 模型 Smoke Test

以下命令仅验证环境和代码链路，不可作为最终报告实验：

```bash
python train_custom.py --set training.epochs=1 --set data.batch_size=4 --set output.experiment_name=smoke_custom
python train_resnet.py --set training.epochs=1 --set data.batch_size=4 --set output.experiment_name=smoke_resnet
python train_eval_strong.py --set training.epochs=1 --set data.batch_size=4 --set output.experiment_name=smoke_convnext
```

每条命令的通过标准：

- [ ] 模型可构建；
- [ ] DataLoader 可读取图像；
- [ ] 至少跑完一个 train batch 和一个 validation batch；
- [ ] `history.csv` 存在；
- [ ] `resolved_config.yaml` 存在；
- [ ] `best_val_raw.pt` 存在；
- [ ] EMA 配置额外生成 `best_val_ema.pt`；
- [ ] 显存没有持续增长或 OOM。

若 ResNet 448×448 smoke test OOM，可临时添加：

```bash
--set preprocess.image_size=224 --set preprocess.resize_size=256
```

正式实验是否恢复 448×448，必须以 validation 对比作出说明。

## E. Checkpoint

对每个候选模型执行：

```bash
python evaluate.py --checkpoint <checkpoint> --split val --output_dir checkpoint_precheck
```

- [ ] checkpoint 可在全新 Python 进程加载；
- [ ] 不需要手动指定模型架构、类别数、图像尺寸或归一化；
- [ ] 生成 `metrics.json`、`predictions.csv`、`per_class_metrics.csv` 和混淆矩阵；
- [ ] `metrics.json` 与 checkpoint 的 validation 指标在合理浮点误差内一致；
- [ ] `best_val_ema.pt` 真正使用 EMA 权重；
- [ ] 开启 `--tta` 时结果目录/记录明确标记 TTA。

旧版 `.pth` 只能通过 `migrate_legacy_checkpoint.py` 迁移成 `legacy_unverified` raw 权重，不能重建旧训练时没有保存的 EMA shadow。

## F. 正式实验

### F1. 模型对照

- [ ] Custom CNN 正式训练完成；
- [ ] ResNet-50 正式训练完成；
- [ ] ConvNeXt-Tiny baseline 正式训练完成；
- [ ] ConvNeXt-Attention 正式训练完成；
- [ ] 每次运行有独立 timestamp 目录；
- [ ] 每个运行保留 config、history 和 checkpoint；
- [ ] 选模型只依据 validation。

### F2. 最小消融

- [ ] A：ConvNeXt baseline；
- [ ] B：+ Attention；
- [ ] C：+ EMA；
- [ ] D：C checkpoint + TTA；
- [ ] 每项都有对应 checkpoint 或 evaluation output；
- [ ] 除目标组件外，数据、seed、epoch、输入尺寸和主要训练参数一致。

### F3. Official test

最终模型冻结后才允许：

```bash
python evaluate.py --checkpoint <final_checkpoint> --split test --output_dir <final_test_dir>
python evaluate.py --checkpoint <final_checkpoint> --split test --tta --output_dir <final_test_tta_dir>
```

- [ ] 不根据 test 结果继续改模型或超参数；
- [ ] 报告 checkpoint 是 raw 还是 EMA；
- [ ] 报告是否开启 TTA；
- [ ] 记录 Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1、Top-3 Accuracy；
- [ ] 保存完整预测、每类指标和混淆矩阵；
- [ ] 报告中的每个数字与 `metrics.json` 完全一致。

## G. 分析与展示

- [ ] 绘制最终模型训练曲线；
- [ ] 导出 Top-10 易混淆类别对；
- [ ] 展示至少 5 个正确样例和 5 个错误样例；
- [ ] 展示至少 3 张 Grad-CAM；
- [ ] 解释背景、遮挡、姿态、相似纹理或样本质量造成的失败；
- [ ] 在 `docs/model_card.md` 填入新版实际指标和限制；
- [ ] README 首屏加入最终模型对比表和关键图。

## H. 课程现场 Demo

- [ ] 用仅含少数类别的模拟图片集测试 `demo_batch.py`；
- [ ] `labels.csv` 使用 `image,class_name` 格式；
- [ ] 输出每张图的真实标签、预测标签、Top-3 与置信度；
- [ ] 输出 Accuracy、Precision、Recall、F1 和混淆矩阵；
- [ ] Demo 输出目录不覆盖正式 test 结果；
- [ ] 可在 30 分钟内完成图片放置、命令执行和结果解释。

## I. 作品集 Demo

- [ ] `python app.py --checkpoint <checkpoint>` 可启动；
- [ ] 上传图片能输出 Top-3；
- [ ] 与批量 evaluator 对同一图片的预测一致；
- [ ] Grad-CAM 正常生成；
- [ ] 页面明确模型名称、数据范围与局限；
- [ ] 默认不使用 `--share`，仅本地启动。

## J. Git 与发布

- [ ] Git 仓库不含 CUB 原图；
- [ ] Git 仓库不含大 checkpoint、训练缓存和临时输出；
- [ ] README 中数据路径与每条命令正确；
- [ ] 在干净 Conda 环境从 README 复现 smoke test；
- [ ] release note 记录最终 checkpoint 路径、hash、配置和 final test 输出位置；
- [ ] 项目主页展示模型比较表、Grad-CAM 和错误分析图。

## Portfolio-ready 判定

只有满足以下条件，项目才标记为 `portfolio-ready`：

1. A–E 全部通过；
2. F1 完成三类模型，F2 至少完成四组最小消融；
3. F3 的 final test 在配置冻结后运行；
4. G、H、I 各有真实输出；
5. J 的干净环境 smoke test 成功。

