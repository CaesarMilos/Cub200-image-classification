# Rewrite changelog

## Version 2 rewrite

- Replaced destructive copied ImageFolder splits with deterministic CSV manifests.
- Added fixed class-map persistence and arbitrary labelled demo dataset support.
- Added a coursework-compliant Custom Fine-Grained CNN.
- Centralised ResNet-50 and ConvNeXt model construction.
- Removed the ineffective ConvNeXt `--model resnet50` branch.
- Added separate best raw and best EMA validation checkpoints.
- Added self-describing checkpoint metadata and legacy raw checkpoint migration.
- Separated training, official evaluation, batch demo, and Gradio inference.
- Added Macro/Weighted metrics, Top-k accuracy, per-class metrics, prediction CSVs, confusion analysis, and Grad-CAM.
- Added YAML configs, timestamped run directories, tests, model card, experiment protocol, and demo guide.

