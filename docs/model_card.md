# Model Card Template

> 在新版训练完成后填写，不要提前复制旧版指标。

## Model

- Name:
- Backbone:
- Custom head:
- Input resolution:
- Selected weights: raw / EMA
- TTA: enabled / disabled

## Data

- Dataset: CUB-200-2011
- Official train images: 5,994
- Official test images: 5,794
- Train/validation method: class-stratified 90/10, seed=42

## Final metrics

| Metric | Validation | Official test |
|---|---:|---:|
| Accuracy | | |
| Macro Precision | | |
| Macro Recall | | |
| Macro F1 | | |
| Weighted F1 | | |
| Top-3 Accuracy | | |

## Intended use

用于 CUB-200-2011 学术数据集上的细粒度鸟类分类、课程演示和个人作品集展示，不应直接作为野外鸟类保护或生态决策工具。

## Known limitations

- 仅覆盖 CUB 定义的 200 类；
- 对遮挡、极端姿态、分布外图片和背景偏差可能敏感；
- Grad-CAM 是事后解释方法，不等价于因果解释。

