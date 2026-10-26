"""Build a supervised dataset (raw / background-subtracted / cleaned) from frames."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .background import estimate_background_from_paths, subtract_background
from .cleaning import CleanConfig, clean_image
from .io_utils import list_images, load_gray, save_gray

__all__ = ["DatasetStats", "build_dataset"]


@dataclass
class DatasetStats:
    frames: int
    background_pixels: int
    background_share: float
    mean_energy_in: float
    mean_energy_bg: float
    mean_energy_out: float


def build_dataset(
    frames_dir: str,
    output_dir: str,
    *,
    background: np.ndarray | None = None,
    batch_size: int = 200,
    brightness_threshold: int = 1,
    std_threshold: float = 50.0,
    config: CleanConfig | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> DatasetStats:
    """Create ``input/``, ``bg_subtracted/`` and ``target/`` image folders.

    ``input`` holds the raw frames, ``bg_subtracted`` the frames after static
    background subtraction and ``target`` the fully cleaned frames.  The pair
    ``input -> target`` is what the (unsuccessful) U-Net experiment was trained on.
    """
    paths = list_images(frames_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No images found in {frames_dir}")

    if background is None:
        background, _ = estimate_background_from_paths(
            paths,
            batch_size=batch_size,
            brightness_threshold=brightness_threshold,
            std_threshold=std_threshold,
        )

    dirs = {
        "input": os.path.join(output_dir, "input"),
        "bg_subtracted": os.path.join(output_dir, "bg_subtracted"),
        "target": os.path.join(output_dir, "target"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    energy_in = energy_bg = energy_out = 0.0
    for i, path in enumerate(paths, start=1):
        name = os.path.basename(path)
        frame = load_gray(path)
        bg_subtracted = subtract_background(frame, background)
        target = clean_image(bg_subtracted, config)

        save_gray(frame, os.path.join(dirs["input"], name))
        save_gray(bg_subtracted, os.path.join(dirs["bg_subtracted"], name))
        save_gray(target, os.path.join(dirs["target"], name))

        energy_in += float(frame.sum())
        energy_bg += float(bg_subtracted.sum())
        energy_out += float(target.sum())

        if verbose and i % 20 == 0:
            print(f"  processed {i}/{len(paths)}")

    bg_pixels = int((background > brightness_threshold).sum())
    stats = DatasetStats(
        frames=len(paths),
        background_pixels=bg_pixels,
        background_share=bg_pixels / background.size,
        mean_energy_in=energy_in / len(paths),
        mean_energy_bg=energy_bg / len(paths),
        mean_energy_out=energy_out / len(paths),
    )

    if verbose:
        print(
            f"Dataset ready: {stats.frames} frames, "
            f"background covers {stats.background_share:.1%} of the sensor, "
            f"mean energy {stats.mean_energy_in:.0f} -> {stats.mean_energy_bg:.0f} -> "
            f"{stats.mean_energy_out:.0f}"
        )
    return stats
