"""Shared builders for the tests: flat synthetic colour images."""
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

NEUTRAL = {"contrast": 0.0, "vibrance": 0.0, "warmth": 0.0}


def hsv_image(h_deg: float, s: float, v: float, size=(120, 90)) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(h_deg / 360, s, v)
    return np.broadcast_to(np.array([r, g, b], np.float32), (size[1], size[0], 3)).copy()


def save_png(arr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8), "RGB").save(path)
    return path
