# Experiment Log

This file records completed CUB-200-2011 experiments. It separates observed
results from conclusions and keeps the official test set locked until model
selection is complete.

## Dataset Contract

- Dataset: official CUB-200-2011, 11,788 images and 200 classes.
- Fixed split: 5,400 training images, 594 validation images, and 5,794 official
  test images.
- Split seed: 42. The official test split is not used for model selection.

## E1 - Custom CNN v1

| Field | Record |
| --- | --- |
| Status | Completed; validation-only result |
| Date | 2026-07-16 |
| Entry point | `train_custom.py` |
| Configuration | `configs/custom_cnn.yaml` |
| Model | Multi-Scale SE-Residual Custom CNN trained from scratch |
| Training epochs | 120 |
| Run directory | `outputs/custom_cnn_v2/20260716_074735/` |
| Model selection | Validation Accuracy, raw weights |
| Official test evaluated | No |

### Final observed epoch metrics

| Variant | Train Accuracy | Validation Accuracy | Validation Macro F1 |
| --- | ---: | ---: | ---: |
| Raw weights | 99.52% | 36.87% | 34.62% |
| EMA weights | - | 5.05% | 2.61% |

### Interpretation

- The raw model almost memorised the training set while validation performance
  remained around 37%, indicating substantial overfitting.
- EMA with decay `0.9999` did not converge sufficiently for this from-scratch
  run and should not be used for this experiment's demo or final evaluation.
- Preserve `best_val_raw.pt` as the selected Custom CNN checkpoint. Do not run
  official test evaluation until the full model comparison and hyperparameter
  selection process is complete.
- This experiment remains a meaningful from-scratch baseline for comparison
  with ResNet-50 and ConvNeXt transfer-learning models.

## E2 - ResNet-50 Transfer Learning

| Field | Record |
| --- | --- |
| Status | Completed; validation-only result |
| Date | 2026-07-16 |
| Entry point | `train_resnet.py` |
| Configuration | `configs/resnet50.yaml` |
| Model | ImageNet-pretrained ResNet-50 transfer-learning baseline |
| Training epochs | 40 |
| Run directory | `outputs/resnet50_transfer_v2/20260716_080917/` |
| Model selection | Validation Accuracy, raw weights |
| Official test evaluated | No |

### Final observed epoch metrics

| Variant | Train Accuracy | Validation Accuracy | Validation Macro F1 |
| --- | ---: | ---: | ---: |
| Raw weights, epoch 40 | 100.00% | 83.50% | 83.63% |

### Best observed validation point in the terminal log

| Epoch | Validation Accuracy | Validation Macro F1 |
| ---: | ---: | ---: |
| 27 | 85.19% | 84.35% |

### Interpretation

- The transfer-learning baseline substantially outperformed the Custom CNN v1
  validation result, demonstrating the value of ImageNet pretraining on this
  limited fine-grained dataset.
- The final epoch was below the best observed validation point, so the selected
  checkpoint must be `best_val_raw.pt`, not an assumption that the final epoch
  is best.
- Do not run official test evaluation until ConvNeXt experiments and model
  selection are complete.

## E3 - ConvNeXt-Tiny Baseline

| Field | Record |
| --- | --- |
| Status | Pending |
| Entry point | `train_eval_strong.py` |
| Configuration | `configs/convnext_baseline.yaml` |
| Official test evaluated | No |

## E4 - ConvNeXt-Tiny with Attention Pooling and EMA

| Field | Record |
| --- | --- |
| Status | Pending |
| Entry point | `train_eval_strong.py` |
| Configuration | `configs/convnext_attention.yaml` |
| Official test evaluated | No |

## Final Comparison Table

Fill this table only after every selected checkpoint has been evaluated once on
the official test split by `evaluate.py`.

| Model | Selected validation Accuracy | Test Accuracy | Macro F1 | Top-3 Accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Custom CNN | Pending best-checkpoint value | Pending | Pending | Pending | Raw weights only |
| ResNet-50 | Pending | Pending | Pending | Pending | Transfer-learning baseline |
| ConvNeXt-Tiny | Pending | Pending | Pending | Pending | Modern CNN baseline |
| ConvNeXt + Attention + EMA | Pending | Pending | Pending | Pending | Main candidate |
