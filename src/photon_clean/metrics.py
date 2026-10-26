"""Quality metrics used to validate the pipeline.

The original task was scored with MSE and with the ratio of total pixel
intensity (``sum(prediction) / sum(target)``), so both are kept here next to the
usual PSNR/SSIM pair.  For sparse photon images we also count photon events
(connected components), which is far more informative than a global PSNR.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.metrics import structural_similarity as _ssim

__all__ = ["mse", "psnr", "ssim", "photon_intensity_ratio", "event_count"]


def _as_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error between two 8-bit images."""
    return float(mean_squared_error(reference, test))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    return float(peak_signal_noise_ratio(reference, test, data_range=data_range))


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity index."""
    return float(_ssim(reference, test, data_range=data_range))


def photon_intensity_ratio(prediction: np.ndarray, target: np.ndarray) -> float:
    """``sum(prediction) / sum(target)`` -- 1.0 means the total signal is preserved."""
    target_sum = float(_as_float(target).sum())
    if target_sum == 0:
        return float("nan")
    return float(_as_float(prediction).sum()) / target_sum


def event_count(image: np.ndarray, threshold: int = 1, min_area: int = 1) -> int:
    """Number of connected bright blobs (photon events) above ``threshold``."""
    binary = (np.asarray(image) > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas >= min_area).sum())
