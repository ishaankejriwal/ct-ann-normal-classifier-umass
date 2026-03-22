# ANN vs NORMAL Video Classification

PyTorch training pipeline for binary classification of echocardiography video clips:
- NORMAL
- ANN_1_4

The project uses a stabilized EfficientNet + BiLSTM + attention model, scan-assignment-driven cross-validation, balanced sampling, detailed metrics, and optional Grad-CAM diagnostics.

## Repository Layout

- train.py: Entry point to start training.
- ann_normal_training/: Main training package.
  - config.py: Runtime configuration and seed/device helpers.
  - dataset.py: Dataset loading, balancing, and collate helpers.
  - model.py: Model architecture and loss/stability components.
  - evaluation.py: Metrics, thresholding, and Grad-CAM utilities.
  - training.py: Cross-validation orchestration and checkpointing.

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended
- PyTorch ecosystem and scientific Python stack

Install dependencies in your virtual environment:

```powershell
pip install torch torchvision numpy pandas scikit-learn tqdm opencv-python
```

## Data Expectations

The training code expects:
- Metadata CSV: aug.csv
- Scan assignment CSV: Scan Assignments.csv
- Preprocessed frame tensors directory: preprocessed_frames_aug/
- Frame files as .pt tensors with shape (3, 224, 224)

Expected key columns in aug.csv:
- TYPE: class label (ANN_1_4 or NORMAL)
- VIDEO_NAME: source video name
- SCAN: scan-level grouping key for leakage-safe CV
- AUGMENT: augmentation tag (orig, hflip), if available

Expected key columns in Scan Assignments.csv:
- TYPE: class label (ANN_1_4 or NORMAL)
- SCAN: scan ID
- FOLD_GROUP: fold group key (for example A-E)
- VIDEO_COUNT: expected orig-video count for the scan

During startup, training validates that `VIDEO_COUNT` matches the number of `orig` rows in `aug.csv` for each `(TYPE, SCAN)` pair and then builds folds directly from `FOLD_GROUP`.

## Configuration

Main configuration lives in ann_normal_training/config.py under class Config.

Common fields you may want to adjust:
- CSV_PATH
- FRAME_DIR
- START_FOLD
- EPOCHS
- BATCH_SIZE
- INITIAL_LR
- SAVE_BEST_MODELS
- SAVE_METRICS

## Running Training

From the project root:

```powershell
python train.py
```

If using your local venv executable directly:

```powershell
& "C:/Users/ishaa/Downloads/ml project/venv/Scripts/python.exe" "train.py"
```

## Outputs

During and after training, outputs are saved to:
- logs/: timestamped training logs and summary JSON
- saved_models/: best-model checkpoints
- metrics/: per-epoch metric JSON files
- split_summaries/: fold split CSV summaries
- gradcams/: periodic Grad-CAM videos and summaries

## Reproducibility Notes

- Seeds are set in set_global_seed in config.py.
- Group-stratified folds are used to reduce scan leakage.
- Validation is evaluated without test-time augmentation.

## GitHub Notes

Large datasets, model weights, caches, and generated artifacts are excluded via .gitignore.
If you need to share trained weights, publish them as a release asset or external storage link.

## License

Add your preferred license in a LICENSE file before public release.
