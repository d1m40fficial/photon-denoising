"""Photon image cleaning: static-background subtraction and hot-pixel removal.

The package implements the classical computer-vision pipeline that turned out to be
the working solution for noisy photon-counting camera frames:

1. estimate a static (per-pixel) background over a stack of frames;
2. subtract it from every frame;
3. remove isolated hot pixels, hot-pixel clusters and residual noise.

A U-Net baseline trained on the same data is kept in ``notebooks/legacy`` for
reference only; it did not reach usable quality (see README).
"""

from .background import (
    estimate_background,
    estimate_background_from_paths,
    subtract_background,
)
from .cleaning import (
    CleanConfig,
    clean_image,
    remove_anomalous_clusters,
    remove_hot_pixels,
    remove_small_components,
)
from .dataset import build_dataset
from .io_utils import list_images, load_gray, natural_sort_key, save_gray
from .metrics import event_count, mse, photon_intensity_ratio, psnr, ssim

__all__ = [
    "CleanConfig",
    "estimate_background",
    "estimate_background_from_paths",
    "subtract_background",
    "clean_image",
    "remove_hot_pixels",
    "remove_anomalous_clusters",
    "remove_small_components",
    "build_dataset",
    "list_images",
    "load_gray",
    "save_gray",
    "natural_sort_key",
    "mse",
    "psnr",
    "ssim",
    "photon_intensity_ratio",
    "event_count",
]

__version__ = "1.0.0"
