"""Pixel-level cleaning: hot pixels, hot-pixel clusters and residual noise.

All operations are vectorised with OpenCV filters, so a 1360x1040 frame is
processed in a few milliseconds instead of seconds (see ``scripts/benchmark.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import median
from skimage.morphology import disk

__all__ = [
    "CleanConfig",
    "remove_hot_pixels",
    "remove_anomalous_clusters",
    "remove_small_components",
    "clean_image",
]


@dataclass(frozen=True)
class CleanConfig:
    """Parameters of the cleaning pipeline.

    The defaults reproduce the tuned values from the original experiment.
    """

    #: A pixel must be brighter than this to be tested as a hot pixel.
    hot_pixel_intensity: int = 1
    #: Neighbourhood mean below which a bright pixel counts as isolated.
    hot_pixel_neighbour_mean: float = 1.0
    #: Hot-pixel neighbourhood radius (1 -> 3x3 window).
    hot_pixel_radius: int = 1
    #: How many times brighter than its local average a cluster must be.
    cluster_intensity_ratio: float = 4.0
    #: Allowed deviation from the Gaussian-smoothed image to call a cluster uniform.
    cluster_similarity: float = 0.15
    #: Radius of the box filter used to compute the local average.
    cluster_radius: int = 1
    #: Radius of the median filter applied at the end (0 disables it).
    median_radius: int = 1


def _neighbourhood_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[radius, radius] = 0.0
    return kernel


def remove_hot_pixels(
    image: np.ndarray,
    intensity_threshold: int = 1,
    neighbour_mean: float = 1.0,
    radius: int = 1,
) -> np.ndarray:
    """Zero out isolated bright pixels that have no bright neighbours.

    A pixel is removed when it is brighter than ``intensity_threshold``, the mean
    of its non-zero neighbours is below ``neighbour_mean`` and it is more than
    three times brighter than that mean.  This targets single-pixel sensor
    defects, which always look brighter than the (empty) area around them.
    """
    image = np.asarray(image, dtype=np.uint8)
    kernel = _neighbourhood_kernel(radius)

    neighbour_sum = cv2.filter2D(image.astype(np.float32), -1, kernel)
    neighbour_count = cv2.filter2D((image > 0).astype(np.float32), -1, kernel)

    with np.errstate(divide="ignore", invalid="ignore"):
        neighbour_avg = np.where(neighbour_count > 0, neighbour_sum / neighbour_count, 0.0)

    hot = (
        (image > intensity_threshold)
        & (neighbour_avg < neighbour_mean)
        & (image > 3.0 * neighbour_avg)
    )

    cleaned = image.copy()
    cleaned[hot] = 0
    return cleaned


def remove_anomalous_clusters(
    image: np.ndarray,
    intensity_ratio: float = 4.0,
    similarity_threshold: float = 0.15,
    radius: int = 1,
) -> np.ndarray:
    """Zero out flat, uniformly bright clusters (multi-pixel sensor defects).

    A cluster is flagged when the pixel is ``intensity_ratio`` times brighter than
    its local average *and* it is close to the Gaussian-smoothed image, i.e. it is
    a plateau rather than a real photon (photons are sharp, so they are preserved).
    """
    image = np.asarray(image, dtype=np.uint8)
    img = image.astype(np.float32)

    size = 2 * radius + 1
    local_avg = cv2.filter2D(img, -1, np.full((size, size), 1.0 / size**2, dtype=np.float32))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = img / (local_avg + 1e-6)

    is_anomalous = (ratio > intensity_ratio) | (ratio < 1.0 / intensity_ratio)
    smoothed = cv2.GaussianBlur(img, (5, 5), 1.0)
    is_uniform = np.abs(img - smoothed) < similarity_threshold * 255.0

    cleaned = image.copy()
    cleaned[is_anomalous & is_uniform] = 0
    return cleaned


def remove_small_components(image: np.ndarray, min_area: int = 5) -> np.ndarray:
    """Drop connected components (photon candidates) smaller than ``min_area``."""
    image = np.asarray(image, dtype=np.uint8)
    binary = (image > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], image, 0).astype(np.uint8)


def clean_image(image: np.ndarray, config: CleanConfig | None = None) -> np.ndarray:
    """Run the full cleaning chain on a background-subtracted frame."""
    cfg = config or CleanConfig()
    cleaned = remove_hot_pixels(
        image,
        intensity_threshold=cfg.hot_pixel_intensity,
        neighbour_mean=cfg.hot_pixel_neighbour_mean,
        radius=cfg.hot_pixel_radius,
    )
    cleaned = remove_anomalous_clusters(
        cleaned,
        intensity_ratio=cfg.cluster_intensity_ratio,
        similarity_threshold=cfg.cluster_similarity,
        radius=cfg.cluster_radius,
    )
    if cfg.median_radius > 0:
        cleaned = median(cleaned, disk(cfg.median_radius)).astype(np.uint8)
    return cleaned
