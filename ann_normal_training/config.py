# Central configuration for ANN vs NORMAL training.

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    # Store immutable training configuration constants.

    # Data and path configuration.
    CSV_PATH: str = "aug.csv"
    ASSIGNMENTS_CSV_PATH: str = "Scan Assignments.csv"
    FRAME_DIR: str = "preprocessed_frames_aug"
    SAVE_DIR: str = "saved_models"
    LOG_DIR: str = "logs"

    # Cross-validation and data loading settings.
    START_FOLD: int = 1
    EPOCHS: int = 150
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 0
    INCLUDE_CLASSES: tuple[str, str] = ("ANN_1_4", "NORMAL")
    FAIL_ON_COUNT_MISMATCH: bool = True
    USE_HFLIP_FOR_TRAIN: bool = False
    USE_VFLIP_FOR_TRAIN: bool = True

    # Optimization and stability settings.
    PATIENCE: int = 20
    MIN_LR: float = 5e-7
    FIXED_THRESHOLD: float = 0.4
    INITIAL_LR: float = 2e-5
    BACKBONE_LR_SCALE: float = 0.1
    WEIGHT_DECAY: float = 2e-4
    DROPOUT_RATE: float = 0.25
    TEMPERATURE: float = 1.0
    MAX_GRAD_NORM: float = 1.0
    ACCUMULATION_STEPS: int = 2
    MAX_LOGIT: float = 8.0
    WARMUP_EPOCHS: int = 8
    EARLY_STOPPING_PATIENCE: int = 21
    MIN_DELTA: float = 0.005
    SAVE_BEST_MODELS: bool = True
    SAVE_METRICS: bool = True


def get_device() -> torch.device:
    # Return the preferred training device.

    # Use CUDA automatically when available.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = 42) -> None:
    # Set deterministic seeds for reproducible training.

    # Seed core random number generators used in this project.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
