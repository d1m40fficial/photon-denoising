"""Image I/O helpers (8-bit grayscale frames)."""

from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def natural_sort_key(name: str) -> list:
    """Sort ``frame_2`` before ``frame_10`` instead of lexicographically."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", name)]


def list_images(directory: str, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[str]:
    """Return image paths in a directory, naturally sorted."""
    exts = tuple(e.lower() for e in extensions)
    names = [n for n in os.listdir(directory) if n.lower().endswith(exts)]
    names.sort(key=natural_sort_key)
    return [os.path.join(directory, n) for n in names]


def load_gray(path: str) -> np.ndarray:
    """Load an image as a 2-D ``uint8`` array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def save_gray(array: np.ndarray, path: str) -> None:
    """Save a 2-D array as an 8-bit grayscale image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)
