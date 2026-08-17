# 迁移与恢复说明

本压缩包包含当前可访问的完整源代码、配置、测试、文档、最终图文报告和可视化脚本，适合迁移到新笔记本后继续开发、展示或重新生成结果。

## 已包含

- 全部 Python 源码、模型、训练、评估和推理脚本；
- VS Code 一键运行配置；
- `requirements.txt` 与 `environment.yml`；
- 全部 YAML 实验配置和 pytest 测试；
- `docs/final_project_report.md` 最终图文报告；
- `results/experiment_log.md` 实验记录；
- `plot_training_curves.py` 与 `generate_final_visuals.py` 两个图表生成器。

## 当前压缩包中未发现的运行产物

这些内容不在当前可访问的工程目录中，因此无法被诚实地加入本压缩包：

- `data/`：CUB-200-2011 原始数据集；
- `outputs/`：训练 checkpoint、`history.csv`、官方 test 的 `final_test_raw/` 与 `final_test_tta/`；
- `results/figures/*.png`：训练曲线、混淆矩阵、预测案例和 Grad-CAM 图片。

它们通常体积较大，并且本项目的 `.gitignore` 也会刻意忽略这些本地产物。

## 从旧硬盘恢复后应放置的位置

如果旧电脑硬盘还能读取，请优先从旧工程目录复制下列文件夹到本项目根目录：

```text
CUB_200_Classification_v2/
├─ data/
└─ outputs/
```

其中最终展示最关键的是：

```text
outputs/convnext_attention_v2/20260716_093755/
├─ best_val_raw.pt
├─ final_test_raw/
└─ final_test_tta/
```

恢复上述目录后：

1. 检查 `configs/data.yaml` 中 `raw_dir` 是否指向新笔记本上的数据集目录；
2. 在 VS Code 运行 `plot_training_curves.py`，生成训练曲线；
3. 运行 `generate_final_visuals.py`，生成混淆矩阵、预测案例和 Grad-CAM；
4. 用 VS Code 打开 `docs/final_project_report.md`，按 `Ctrl + K`、`V` 查看最终图文报告。

无需重新训练；只要恢复数据集和 `outputs/`，现有冻结 checkpoint 即可重新评估或重新生成全部展示图。

