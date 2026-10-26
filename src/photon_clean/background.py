"""Static background estimation and subtraction.

A photon-counting camera adds a *static* pattern to every frame: a per-pixel offset
that does not change between exposures.  Because photons hit random pixels, the
median over many frames of the same scene is a robust estimate of that offset,
while the per-pixel standard deviation tells us which pixels are stable enough to
be treated as background (and which ones keep firing photons).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

from .io_utils import load_gray

__all__ = [
    "estimate_background",
    "estimate_background_from_paths",
    "background_mask",
    "subtract_background",
    "iter_batches",
]


def iter_batches(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of at most ``batch_size`` items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class _BackgroundAccumulator:
    """Streaming median/std accumulation over batches of frames."""

    def __init__(self, brightness_threshold: int, std_threshold: float) -> None:
        self.brightness_threshold = brightness_threshold
        self.std_threshold = std_threshold
        self._sum: np.ndarray | None = None
        self._count: np.ndarray | None = None
        self._std_sum: np.ndarray | None = None
        self.batches = 0

    def add_batch(self, frames: list[np.ndarray]) -> None:
        stack = np.stack(frames).astype(np.float32)
        if self._sum is None:
            shape = stack.shape[1:]
            self._sum = np.zeros(shape, dtype=np.float64)
            self._count = np.zeros(shape, dtype=np.int32)
            self._std_sum = np.zeros(shape, dtype=np.float32)

        median = np.median(stack, axis=0)
        std = np.std(stack, axis=0)
        stable = (std < self.std_threshold) & (median > self.brightness_threshold)

        self._sum += np.where(stable, median, 0.0)
        self._count += stable.astype(np.int32)
        self._std_sum += std
        self.batches += 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sum is None or self._count is None or self._std_sum is None:
            raise ValueError("No frames were provided to estimate the background.")

        background = np.zeros(self._sum.shape, dtype=np.float64)
        valid = self._count > 0
        background[valid] = self._sum[valid] / self._count[valid]
        background = np.where(background > self.brightness_threshold, background, 0)
        std_map = (self._std_sum / self.batches).astype(np.float32)
        return np.clip(background, 0, 255).astype(np.uint8), std_map


def estimate_background(
    frames: Iterable[np.ndarray],
    batch_size: int = 200,
    brightness_threshold: int = 1,
    std_threshold: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a static background from a stack of grayscale frames.

    Frames are consumed in batches so that a full stack never has to be kept in
    memory.  For every pixel we take the median and the standard deviation inside
    a batch; pixels that are bright (``median > brightness_threshold``) and stable
    (``std < std_threshold``) vote for the background.  Votes are averaged over
    all batches.

    Parameters
    ----------
    frames:
        Iterable of 2-D ``uint8`` arrays, all of the same shape.
    batch_size:
        Number of frames held in memory at once.
    brightness_threshold:
        Minimum median value for a pixel to be considered background at all.
    std_threshold:
        Maximum frame-to-frame standard deviation of a background pixel.

    Returns
    -------
    background:
        ``uint8`` image, zero where no stable background was found.
    std_map:
        Mean per-pixel standard deviation over the processed batches.
    """
    accumulator = _BackgroundAccumulator(brightness_threshold, std_threshold)
    batch: list[np.ndarray] = []

    for frame in frames:
        batch.append(np.asarray(frame, dtype=np.uint8))
        if len(batch) >= batch_size:
            accumulator.add_batch(batch)
            batch = []

    if batch:
        accumulator.add_batch(batch)

    return accumulator.result()


def estimate_background_from_paths(
    paths: Sequence[str],
    batch_size: int = 200,
    brightness_threshold: int = 1,
    std_threshold: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper of :func:`estimate_background` for a list of files."""
    return estimate_background(
        (load_gray(p) for p in paths),
        batch_size=batch_size,
        brightness_threshold=brightness_threshold,
        std_threshold=std_threshold,
    )


def background_mask(background: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Boolean mask of pixels that belong to the static background."""
    return np.asarray(background) > threshold


def subtract_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Subtract a static background from a frame, clipping negatives to zero."""
    if image.shape != background.shape:
        raise ValueError(f"Shape mismatch: frame {image.shape} vs background {background.shape}")
    return np.clip(image.astype(np.float32) - background.astype(np.float32), 0, 255).astype(np.uint8)
