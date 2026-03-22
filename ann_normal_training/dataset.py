# Dataset and dataloader helpers for frame tensor videos.

from __future__ import annotations

import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF


def downsample_to_equal(df_in: pd.DataFrame, positive_label: str, seed: int = 42) -> pd.DataFrame:
    # Downsample classes to equal counts for balanced evaluation.

    # Split rows by class to compute the shared target size.
    ann_df = df_in[df_in["TYPE"] == positive_label]
    norm_df = df_in[df_in["TYPE"] == "NORMAL"]
    target = min(len(ann_df), len(norm_df))
    if target == 0:
        return df_in

    ann_keep = ann_df.sample(n=target, random_state=seed) if len(ann_df) > target else ann_df
    norm_keep = norm_df.sample(n=target, random_state=seed) if len(norm_df) > target else norm_df

    out = pd.concat([ann_keep, norm_keep], axis=0)
    # Shuffle after balancing to avoid class ordering artifacts.
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


class StabilizedDataset(Dataset):
    # Load fixed-size frame tensors and return video-label pairs.

    def __init__(
        self,
        dataframe: pd.DataFrame,
        frame_dir: str,
        label_map: dict[str, int],
        temporal_aug=None,
        is_training: bool = True,
        enable_augmentation: bool = False,
    ) -> None:
        # Keep a contiguous index to avoid mismatches in DataLoader workers.
        self.dataframe = dataframe.reset_index(drop=True)
        self.frame_dir = frame_dir
        self.label_map = label_map
        self.temporal_aug = temporal_aug
        self.is_training = is_training
        self.enable_augmentation = enable_augmentation
        # Cache ImageNet stats for normalization-aware intensity transforms.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self) -> int:
        # Return number of samples in the dataframe.

        return len(self.dataframe)

    def _load_video_tensor(self, video_folder: str) -> torch.Tensor:
        # Load all .pt frame tensors for a single video.

        # Read frame files in sorted order to preserve temporal sequence.
        pt_files = sorted([name for name in os.listdir(video_folder) if name.endswith(".pt")])
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in: {video_folder}")

        frames = []
        for file_name in pt_files:
            pt_path = os.path.join(video_folder, file_name)
            try:
                # Load tensors onto CPU first to keep worker memory predictable.
                frame = torch.load(pt_path, map_location="cpu")
                if frame.shape != (3, 224, 224):
                    raise ValueError(f"Invalid shape {frame.shape} in {pt_path}")
                if torch.isnan(frame).any() or torch.isinf(frame).any():
                    raise ValueError(f"NaN/Inf in {pt_path}")
                frames.append(frame)
            except Exception:
                # Replace unreadable frames with zeros to keep sample shape valid.
                frames.append(torch.zeros(3, 224, 224))

        return torch.stack(frames, dim=0)

    def _apply_video_level_spatial_aug(self, video: torch.Tensor) -> torch.Tensor:
        # Apply one shared spatial transform across all frames.

        # Use one random rotation value for the full sequence.
        if random.random() < 0.35:
            angle = random.uniform(-7.0, 7.0)
            video = TF.rotate(video, angle=angle)
        return video

    def _apply_video_level_intensity_aug(self, video: torch.Tensor) -> torch.Tensor:
        # Apply one shared intensity transform across all frames.

        # Convert normalized tensors back to image space before intensity edits.
        denormalized = (video * self.std + self.mean).clamp(0.0, 1.0)
        if random.random() < 0.30:
            contrast = random.uniform(0.90, 1.10)
            denormalized = TF.adjust_contrast(denormalized, contrast)
        renormalized = (denormalized - self.mean) / self.std
        # Clamp to conservative bounds to avoid extreme values.
        return renormalized.clamp(-5.0, 5.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Return a video tensor and integer class label.

        row = self.dataframe.iloc[idx]
        label = self.label_map.get(row["TYPE"], 0)

        # Build the folder path from the video identifier.
        video_name = row["VIDEO_NAME"].replace(".mp4", "")
        video_folder = os.path.join(self.frame_dir, video_name)
        video = self._load_video_tensor(video_folder)

        if self.is_training and self.enable_augmentation:
            # Apply optional sequence-level augmentations during training only.
            video = self._apply_video_level_spatial_aug(video)
            video = self._apply_video_level_intensity_aug(video)

        if self.temporal_aug:
            video = self.temporal_aug(video)

        if torch.isnan(video).any() or torch.isinf(video).any():
            # Replace invalid tensors with zeros as a final safety guard.
            video = torch.zeros_like(video)

        return video, label


def enhanced_pad_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    # Stack fixed-length videos into a batch tensor.

    # Unzip sample tuples and stack videos on the batch dimension.
    videos, labels = zip(*batch)
    video_batch = torch.stack(videos)
    label_batch = torch.tensor(labels, dtype=torch.float32)
    return video_batch, label_batch


def create_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    # Create inverse-frequency weights for balanced minibatches.

    # Compute class frequencies once and invert per-row weights.
    class_counts = df["TYPE"].value_counts()
    total_samples = len(df)
    weights = []

    for _, row in df.iterrows():
        class_name = row["TYPE"]
        class_count = class_counts[class_name]
        # Scale by class frequency so minority samples are drawn more often.
        weight = total_samples / (len(class_counts) * class_count)
        weights.append(weight)

    return WeightedRandomSampler(weights, len(weights), replacement=True)


def validate_balanced_sampling(
    train_loader: DataLoader,
    logger,
    fold: int,
    positive_label: str = "POSITIVE",
    num_batches_to_check: int = 10,
) -> float:
    # Log class balance over early batches from the training loader.

    # Inspect early batches to verify sampler behavior quickly.
    logger.info(f"[VALIDATION] Checking class balance in first {num_batches_to_check} batches for fold {fold}.")

    total_normal = 0
    total_ann = 0
    for batch_idx, (_, labels) in enumerate(train_loader):
        if batch_idx >= num_batches_to_check:
            break

        batch_normal = (labels == 0).sum().item()
        batch_ann = (labels == 1).sum().item()
        total_normal += batch_normal
        total_ann += batch_ann
        logger.info(f"Batch {batch_idx + 1}: NORMAL={batch_normal}, {positive_label}={batch_ann}")

    total = total_normal + total_ann
    balance_ratio = (total_ann / total) if total > 0 else 0.0
    logger.info(f"[VALIDATION] {positive_label} ratio: {balance_ratio:.3f} (target ~0.5).")
    return balance_ratio
