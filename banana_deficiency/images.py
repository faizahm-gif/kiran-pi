"""Image reading that keeps Unicode Windows paths intact."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def open_bgr(path: Path) -> np.ndarray:
    rgb = np.asarray(open_rgb(path))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
