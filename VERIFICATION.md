# Verification report

## Completed in the delivery environment

- All Python files pass `python -m compileall` syntax compilation.
- Every Python module has a module-level purpose docstring.
- Every Python class has a class-level function docstring.
- All six YAML files parse successfully.
- Every experiment config contains the required data/model/preprocess/training/evaluation/output sections.
- The rewritten metadata pipeline was executed against the uploaded CUB official metadata:
  - train: 5,400 images;
  - validation: 594 images;
  - official test: 5,794 images;
  - classes: 200;
  - split image IDs are disjoint;
  - generated test IDs exactly match the official test split;
  - manifest and class-map JSON round trips succeed.

## Requires the user's local PyTorch environment

The delivery container does not include PyTorch, TorchVision, or Gradio. The following checks must therefore be run in the user's `py311`/RTX 5090 environment before full training:

```bash
pytest -q
python train_custom.py --set training.epochs=1 --set data.batch_size=4
python train_resnet.py --set training.epochs=1 --set data.batch_size=4
python train_eval_strong.py --set training.epochs=1 --set data.batch_size=4
```

These smoke runs should use a dedicated output experiment name or temporary output directory; they are not reportable final experiments.

