# Training orchestration for cross-validated ANN vs NORMAL modeling.

from __future__ import annotations

import gc
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, get_device, set_global_seed
from .dataset import StabilizedDataset, create_balanced_sampler, downsample_to_equal, enhanced_pad_collate, validate_balanced_sampling
from .evaluation import (
    calculate_detailed_metrics,
    evaluate_without_tta,
    find_optimal_threshold_conservative,
    generate_gradcam_for_tensor,
    print_detailed_metrics,
    save_gradcam_video,
)
from .logging_utils import log_complete, log_error, log_save, log_start, log_stats, log_stop, log_success, log_target, setup_logging
from .model import GradientStabilizer, StabilizedCNNLSTM, StabilizedFocalLoss


class StabilizedScheduler:
    # Apply linear warmup followed by cosine learning-rate decay.

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.group_base_lrs = [param_group.get("lr", base_lr) for param_group in self.optimizer.param_groups]

    def step(self, epoch: int) -> float:
        # Update optimizer learning rate for the current epoch.

        # Use linear warmup before cosine decay.
        if epoch < self.warmup_epochs:
            learning_rates = [
                self.min_lr + (group_base_lr - self.min_lr) * (epoch + 1) / self.warmup_epochs
                for group_base_lr in self.group_base_lrs
            ]
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_scale = 0.5 * (1 + np.cos(np.pi * progress))
            learning_rates = [
                self.min_lr + (group_base_lr - self.min_lr) * cosine_scale
                for group_base_lr in self.group_base_lrs
            ]

        for param_group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
            param_group["lr"] = learning_rate
        return float(learning_rates[0])


class QualityEarlyStopping:
    # Stop training when quality-gated F1 stops improving.

    def __init__(self, patience: int = 15, min_precision: float = 0.5, min_specificity: float = 0.2, min_delta: float = 0.002) -> None:
        self.patience = patience
        self.min_precision = min_precision
        self.min_specificity = min_specificity
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.patience_counter = 0

    def __call__(self, f1_score: float, precision_score: float, _auc_score: float, specificity: float) -> tuple[bool, bool]:
        # Return stop/save flags based on quality and improvement gates.

        if precision_score >= self.min_precision and specificity >= self.min_specificity:
            if f1_score > self.best_f1 + self.min_delta:
                self.best_f1 = f1_score
                self.patience_counter = 0
                return False, True
            self.patience_counter += 1
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            return True, False
        return False, False


def make_json_safe(obj):
    # Convert nested objects to JSON-serializable values.

    # Recursively convert common NumPy and tensor types.
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_best_model(model, optimizer, scheduler, metrics: dict, fold: int, epoch: int, save_dir: str = "saved_models") -> str:
    # Persist a full checkpoint for the current best model.

    # Create output folder and timestamped checkpoint metadata.
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.__dict__,
        "metrics": metrics,
        "config": {
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.INITIAL_LR,
            "dropout_rate": Config.DROPOUT_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "epochs": Config.EPOCHS,
            "patience": Config.PATIENCE,
        },
        "timestamp": timestamp,
    }

    filename = f"best_model_fold{fold}_epoch{epoch}_f1{metrics['f1_score']:.4f}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


def save_metrics_to_json(metrics: dict, fold: int, epoch: int, save_dir: str = "metrics") -> str | None:
    # Write epoch metrics to JSON for offline analysis.

    # Serialize scalar-like values to plain Python floats.
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"metrics_fold{fold}_epoch{epoch}_{timestamp}.json")

    serializable_metrics = {}
    for key, value in metrics.items():
        if key in ["confusion_matrix", "raw_confusion_matrix"]:
            serializable_metrics[key] = value
        elif hasattr(value, "item"):
            serializable_metrics[key] = float(value.item())
        else:
            serializable_metrics[key] = float(value)

    payload = {"fold": fold, "epoch": epoch, "timestamp": timestamp, "metrics": serializable_metrics}

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return filepath
    except TypeError:
        return None


def make_sgkf_folds_close_to_20(
    df: pd.DataFrame,
    label_col: str = "TYPE",
    group_col: str = "SCAN",
    n_splits: int = 5,
    seeds=range(1, 400),
    tol: float = 0.04,
):
    # Find group-stratified folds with validation share near 20%.

    # Search multiple seeds to get near-uniform validation share.
    target = 1.0 / n_splits
    x_values = df.index.values
    y_values = df[label_col].values
    groups = df[group_col].values

    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(sgkf.split(x_values, y_values, groups))
        shares = [len(val_idx) / len(df) for _, val_idx in folds]
        if all(abs(share - target) <= tol for share in shares):
            return folds, seed, shares

    fallback_seed = list(seeds)[0]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=fallback_seed)
    folds = list(sgkf.split(x_values, y_values, groups))
    shares = [len(val_idx) / len(df) for _, val_idx in folds]
    return folds, fallback_seed, shares


def assert_and_log_fold(df: pd.DataFrame, train_idx, val_idx, fold: int, label_col: str = "TYPE", group_col: str = "SCAN", logger=None) -> None:
    # Validate scan-level separation and log fold composition.

    # Enforce zero overlap between train and validation scan IDs.
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_scans, val_scans = set(train_df[group_col]), set(val_df[group_col])
    overlap = train_scans & val_scans
    assert len(overlap) == 0, f"[LEAKAGE] Fold {fold}: scans overlap: {overlap}"

    val_share = len(val_df) / len(df)
    val_counts = val_df[label_col].value_counts().to_dict()
    ann_count = val_counts.get("NODULE", 0)
    ann_ratio = ann_count / max(1, len(val_df))

    message = (
        f"Fold {fold} | Val videos={len(val_df)} ({val_share:.2%}) | "
        f"ANN in val={ann_count}/{len(val_df)} ({ann_ratio:.2f}) | "
        f"Unique val scans={len(val_scans)}"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def stabilized_training_loop(model, train_loader, val_loader, device: torch.device, logger, fold: int) -> tuple[float, dict | None]:
    # Train one fold with stabilization, logging, and model checkpointing.

    # Configure criterion, optimizer, scheduler, and early stopping.
    criterion = StabilizedFocalLoss(
        alpha=0.48,
        gamma=1.0,
        label_smoothing=0.02,
        temperature=Config.TEMPERATURE,
        max_logit=Config.MAX_LOGIT,
    )

    optimizer = optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in model.cnn.named_parameters() if parameter.requires_grad],
                "lr": Config.INITIAL_LR * 0.1,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": list(model.feature_norm.parameters())
                + list(model.feature_projection.parameters())
                + list(model.lstm.parameters())
                + list(model.self_attention.parameters())
                + list(model.classifier.parameters()),
                "lr": Config.INITIAL_LR,
                "weight_decay": Config.WEIGHT_DECAY,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = StabilizedScheduler(
        optimizer=optimizer,
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.EPOCHS,
        base_lr=Config.INITIAL_LR,
        min_lr=Config.MIN_LR,
    )

    early_stopping = QualityEarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_precision=0.35,
        min_specificity=0.35,
        min_delta=Config.MIN_DELTA,
    )

    grad_stabilizer = GradientStabilizer(model, max_norm=Config.MAX_GRAD_NORM)
    scaler = GradScaler()

    best_f1 = 0.0
    best_metrics = None
    best_val_preds = None
    best_val_labels = None

    for epoch in range(Config.EPOCHS):
        # Reset per-epoch accumulators and switch to training mode.
        model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        current_lr = scheduler.step(epoch)
        progress_bar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch + 1}/{Config.EPOCHS} (LR: {current_lr:.2e})")

        for batch_idx, (videos, labels) in enumerate(progress_bar):
            # Move the current batch to the active device.
            videos, labels = videos.to(device), labels.float().to(device)

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                if torch.isnan(videos).any() or torch.isinf(videos).any():
                    logger.warning(f"Corrupt input detected in batch {batch_idx}. Skipping.")
                    continue

                outputs = model(videos)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(f"NaN in model outputs in batch {batch_idx}. Skipping.")
                    continue

                loss = criterion(outputs, labels) / Config.ACCUMULATION_STEPS
                if torch.isnan(loss).any():
                    logger.warning(f"NaN in loss in batch {batch_idx}. Skipping.")
                    continue

            scaler.scale(loss).backward()
            # Accumulate gradients to emulate a larger batch size.
            accumulated_loss += float(loss.item())

            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                logit_stats = criterion.get_logit_stats()
                is_unstable, reason = grad_stabilizer.check_and_clip_gradients(logit_stats)

                if is_unstable:
                    logger.warning(f"Gradient instability detected: {reason}. Skipping step.")
                    optimizer.zero_grad()
                    scaler.update()
                    accumulated_loss = 0.0
                    continue

                scaler.step(optimizer)
                scaler.update()
                # Clear gradients after each optimizer update.
                optimizer.zero_grad()

                total_loss += accumulated_loss
                num_batches += 1
                accumulated_loss = 0.0

                postfix = {"loss": f"{loss.item():.4f}"}
                if logit_stats:
                    postfix["logit_max"] = f"{logit_stats['max']:.2f}"
                    postfix["logit_std"] = f"{logit_stats['std']:.2f}"
                progress_bar.set_postfix(postfix)

        criterion.update_epoch()

        if (epoch + 1) % 5 == 0:
            logit_stats = criterion.get_logit_stats()
            if logit_stats:
                logger.info(
                    f"Fold {fold} Epoch {epoch + 1} Logit Stats - "
                    f"Min: {logit_stats['min']:.3f}, Max: {logit_stats['max']:.3f}, "
                    f"Mean: {logit_stats['mean']:.3f}, Std: {logit_stats['std']:.3f}"
                )

        if num_batches > 0:
            # Run validation and evaluate both fixed and adaptive thresholds.
            avg_loss = total_loss / num_batches
            val_preds, val_labels, _ = evaluate_without_tta(model, val_loader, device)
            metrics_fixed = calculate_detailed_metrics(val_labels, val_preds, threshold=Config.FIXED_THRESHOLD)
            metrics_opt = calculate_detailed_metrics(val_labels, val_preds, threshold=None)

            if epoch % 5 == 0:
                # Generate Grad-CAM diagnostics periodically for error analysis.
                threshold = metrics_fixed["threshold"]
                y_true = np.array(val_labels)
                y_prob = np.array(val_preds)
                y_pred = (y_prob > threshold).astype(int)

                pairs = list(zip(y_true, y_pred, y_prob, val_loader.dataset.dataframe["VIDEO_NAME"]))
                for true_label, pred_label, prob, video_name in pairs:
                    is_fp = pred_label == 1 and true_label == 0
                    is_fn = pred_label == 0 and true_label == 1
                    is_tp = pred_label == 1 and true_label == 1
                    if not (is_fp or is_fn or is_tp):
                        continue

                    label_str = "TP" if is_tp else "FP" if is_fp else "FN"
                    video_folder = os.path.join(Config.FRAME_DIR, str(video_name).replace(".mp4", ""))
                    if not os.path.exists(video_folder):
                        continue

                    frame_paths = sorted([name for name in os.listdir(video_folder) if name.endswith(".pt")])
                    if not frame_paths:
                        continue

                    video_tensor = torch.stack([torch.load(os.path.join(video_folder, frame_name), map_location="cpu") for frame_name in frame_paths])
                    heatmaps = generate_gradcam_for_tensor(model, video_tensor, device)

                    gradcam_dir = os.path.join("gradcams", f"fold{fold}_epoch{epoch + 1}", label_str)
                    output_path = os.path.join(gradcam_dir, f"{video_name}.mp4")
                    save_gradcam_video(video_tensor, heatmaps, output_path, fps=5)

                    summary_path = os.path.join("gradcams", f"fold{fold}_epoch{epoch + 1}", "summary.csv")
                    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                    with open(summary_path, "a", encoding="utf-8") as summary_file:
                        summary_file.write(f"{video_name},{true_label},{pred_label},{prob:.4f},{label_str}\n")

            logger.info(f"Fold {fold} Epoch {epoch + 1}: Loss: {avg_loss:.4f}")
            print("\n--- OPTIMIZED THRESHOLD ---")
            print_detailed_metrics(metrics_opt, fold=fold, epoch=epoch + 1)

            if (epoch + 1) % 5 == 0:
                print("\n--- FIXED THRESHOLD ---")
                print_detailed_metrics(metrics_fixed, fold=fold, epoch=epoch + 1)

            if Config.SAVE_METRICS:
                save_metrics_to_json(metrics_opt, fold, epoch + 1)

            if metrics_opt["f1_score"] > best_f1:
                # Track best validation state for final threshold reporting.
                best_f1 = float(metrics_opt["f1_score"])
                best_metrics = metrics_opt
                best_val_preds = val_preds
                best_val_labels = val_labels

            should_stop, should_save = early_stopping(
                metrics_opt["f1_score"],
                metrics_opt["precision"],
                metrics_opt["auc"],
                metrics_opt["specificity"],
            )

            if should_save and Config.SAVE_BEST_MODELS:
                model_path = save_best_model(model, optimizer, scheduler, metrics_opt, fold, epoch + 1)
                log_save(logger, f"Best model saved: {model_path}")

            if should_stop:
                log_stop(logger, f"Early stopping at epoch {epoch + 1}")
                break

            if not criterion.is_loss_stable():
                logger.warning("Loss instability detected. Consider reducing learning rate.")

    if best_val_preds is not None and best_val_labels is not None:
        final_opt_thr = find_optimal_threshold_conservative(best_val_labels, best_val_preds)
        logger.info(f"[FINAL] One-time optimal threshold (best epoch): {final_opt_thr:.4f}")

    return best_f1, best_metrics


def run_cross_validation() -> None:
    # Run the full cross-validation training pipeline.

    # Initialize logger and emit static run settings.
    logger = setup_logging(Config.LOG_DIR)
    log_start(logger, "Starting stabilized CNN-LSTM training with balanced classes")

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("- Augmentations: dynamic train-time sequence aug disabled; using pre-generated variants only")
    logger.info("- Training sampling: balanced via WeightedRandomSampler")
    logger.info("- Validation sampling: balanced via downsampling")
    logger.info("- Test-time augmentation: disabled")
    logger.info("=" * 60)

    set_global_seed(42)
    # Resolve runtime device once for the full run.
    device = get_device()

    try:
        # Load metadata table and keep only target classes.
        df = pd.read_csv(Config.CSV_PATH)
        df = df[df["TYPE"].isin(Config.INCLUDE_CLASSES)].reset_index(drop=True)
        log_success(logger, f"Data loaded successfully: {len(df)} samples")
    except Exception as error:
        log_error(logger, f"Error loading data: {error}")
        return

    logger.info("Dataset statistics")
    logger.info(df["TYPE"].value_counts().to_dict())
    logger.info(f"Total samples: {len(df)}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    label_map = {"NORMAL": 0, "NODULE": 1}

    df_orig = df[df["AUGMENT"] == "orig"].reset_index(drop=True) if "AUGMENT" in df.columns else df.copy()

    folds, chosen_seed, shares = make_sgkf_folds_close_to_20(df_orig, label_col="TYPE", group_col="SCAN")
    logger.info(f"[SPLIT] Chosen seed={chosen_seed}, val shares per fold={shares}")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        # Respect configured starting fold to support resumed runs.
        if fold_idx < Config.START_FOLD:
            logger.info(f"[SKIP] Skipping fold {fold_idx} (START_FOLD={Config.START_FOLD})")
            continue

        assert_and_log_fold(df_orig, train_idx, val_idx, fold_idx, logger=logger)

        train_orig = df_orig.iloc[train_idx].copy()
        val_orig = df_orig.iloc[val_idx].copy()

        logger.info(f"\nFold {fold_idx}")
        logger.info(f"Train (orig split): {train_orig['TYPE'].value_counts().to_dict()}")
        logger.info(f"Val (orig split): {val_orig['TYPE'].value_counts().to_dict()}")

        train_scans = set(train_orig["SCAN"])
        val_scans = set(val_orig["SCAN"])

        if "AUGMENT" in df.columns:
            # Use orig+hflip for train and orig-only for validation.
            train_df = df[(df["SCAN"].isin(train_scans)) & (df["AUGMENT"].isin(["orig", "hflip"]))].reset_index(drop=True)
            val_df = df[(df["SCAN"].isin(val_scans)) & (df["AUGMENT"] == "orig")].reset_index(drop=True)
        else:
            train_df = train_orig.reset_index(drop=True)
            val_df = val_orig.reset_index(drop=True)

        train_sampler = create_balanced_sampler(train_df)
        # Downsample validation set to equal class counts.
        val_df = downsample_to_equal(val_df, seed=4242)

        logger.info(f"Train (balanced sampling): {train_df['TYPE'].value_counts().to_dict()} (n={len(train_df)})")
        logger.info(f"Val (balanced downsample): {val_df['TYPE'].value_counts().to_dict()} (n={len(val_df)})")

        os.makedirs("split_summaries", exist_ok=True)
        val_df[["VIDEO_NAME", "SCAN", "TYPE"]].assign(fold=fold_idx).to_csv(
            os.path.join("split_summaries", f"val_split_fold{fold_idx}.csv"),
            index=False,
        )

        try:
            # Build dataset objects with shared label mapping.
            train_ds = StabilizedDataset(
                train_df,
                frame_dir=Config.FRAME_DIR,
                label_map=label_map,
                is_training=True,
                enable_augmentation=False,
            )
            val_ds = StabilizedDataset(val_df, frame_dir=Config.FRAME_DIR, label_map=label_map, is_training=False)
            log_success(logger, "Datasets created successfully")
        except Exception as error:
            log_error(logger, f"Error creating datasets: {error}")
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=enhanced_pad_collate,
        )
        validate_balanced_sampling(train_loader, logger, fold_idx, num_batches_to_check=10)

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=enhanced_pad_collate,
        )

        model = StabilizedCNNLSTM(dropout_rate=Config.DROPOUT_RATE).to(device)
        log_target(logger, f"Starting stabilized training for fold {fold_idx}")

        best_f1, best_metrics = stabilized_training_loop(model, train_loader, val_loader, device, logger, fold_idx)

        fold_results.append(
            {
                "fold": fold_idx,
                "f1_score": float(best_f1),
                "best_metrics": make_json_safe(best_metrics),
                "train_samples": int(len(train_df)),
                "val_samples": int(len(val_df)),
            }
        )
        log_success(logger, f"Fold {fold_idx} completed. Best F1: {best_f1:.4f}")

        del model, train_loader, val_loader, train_ds, val_ds
        # Release memory between folds to avoid fragmentation.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if fold_results:
        # Summarize fold-level performance and persist aggregate results.
        logger.info("\n" + "=" * 60)
        log_complete(logger, "CROSS-VALIDATION RESULTS")
        logger.info("=" * 60)

        f1_scores = [result["f1_score"] for result in fold_results]
        for result in fold_results:
            logger.info(f"Fold {result['fold']}: F1={result['f1_score']:.4f}")
            if result["best_metrics"]:
                logger.info(f"  - AUC: {result['best_metrics']['auc']:.4f}")
                logger.info(f"  - Precision: {result['best_metrics']['precision']:.4f}")
                logger.info(f"  - Recall: {result['best_metrics']['recall']:.4f}")

        log_stats(logger, "SUMMARY STATISTICS")
        logger.info(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        logger.info(f"Best F1: {np.max(f1_scores):.4f}")
        logger.info(f"Worst F1: {np.min(f1_scores):.4f}")

        final_results = {
            "fold_results": fold_results,
            "summary": {
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "best_f1": float(np.max(f1_scores)),
                "worst_f1": float(np.min(f1_scores)),
            },
            "timestamp": datetime.now().isoformat(),
        }

        results_file = os.path.join(Config.LOG_DIR, f"stabilized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, "w", encoding="utf-8") as file:
            json.dump(final_results, file, indent=2)
        log_save(logger, f"Final results saved to: {results_file}")

        all_val_csv = os.path.join("split_summaries", "val_splits_all_folds.csv")
        val_files = [
            pd.read_csv(os.path.join("split_summaries", file_name))
            for file_name in sorted(os.listdir("split_summaries"))
            if file_name.startswith("val_split_fold")
        ]
        if val_files:
            pd.concat(val_files).to_csv(all_val_csv, index=False)
            log_save(logger, f"Validation split summary saved to: {all_val_csv}")
    else:
        log_error(logger, "No folds were completed successfully.")

    log_target(logger, "Training completed successfully")
