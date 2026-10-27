"""DINOv3 ViT backbone plus the deficiency classification head.

This module only rebuilds the network from the saved checkpoint. It does not
download pretrained weights and it does not train.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel


class DINOv3Classifier(nn.Module):
    """CLS token concatenated with the mean patch token, then a small MLP."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(pixel_values))

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        cls = tokens[:, 0]
        registers = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
        patches = tokens[:, 1 + registers :, :]
        if patches.shape[1] == 0:
            patch_mean = cls
        else:
            patch_mean = patches.mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)


def load_checkpoint(directory: Path, device: torch.device):
    directory = Path(directory)
    if directory.is_file():
        raise SystemExit(
            "Pass the checkpoint folder that contains model.pt, not the weight file alone. "
            f"Got: {directory}"
        )
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"No checkpoint at {directory}. Expected model.pt, meta.json, processor/, and backbone_config/.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    try:
        config = AutoConfig.from_pretrained(directory / "backbone_config")
        backbone = AutoModel.from_config(config)
    except Exception as exc:
        raise SystemExit(
            "Could not rebuild DINOv3 from the saved config. Install transformers>=4.56 and retry.\n"
            f"{exc}"
        ) from exc
    model = DINOv3Classifier(backbone, num_classes=len(classes), dropout=float(meta["dropout"]))
    try:
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(directory / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    processor = AutoImageProcessor.from_pretrained(directory / "processor")
    model.to(device)
    model.eval()
    return model, processor, meta
