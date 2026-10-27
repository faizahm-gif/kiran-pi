"""Paths and device helpers used at inference time."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAM_WEIGHTS = "sam2.1_b.pt"


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
