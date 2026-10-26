"""Render every figure used in the README into ``assets/``.

    python scripts/make_figures.py

Requires ``results/metrics.json`` (run ``scripts/evaluate.py`` first) and
optionally ``results/benchmark.json`` (``scripts/benchmark.py``).
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk

from _bootstrap import ASSETS, DATA, LEGACY, RESULTS

from photon_clean import (
    event_count,
    list_images,
    load_gray,
    remove_anomalous_clusters,
    remove_hot_pixels,
    subtract_background,
)

INK = "#1f2328"
MUTED = "#59636e"
GRID = "#d1d9e0"
BLUE = "#0969da"
ORANGE = "#bc4c00"
GREEN = "#1a7f37"
RED = "#cf222e"
PURPLE = "#8250df"

CROP = (slice(340, 584), slice(500, 820))


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def _frame(
    ax,
    image: np.ndarray,
    title: str,
    vmax: float = 30.0,
    crop: bool = False,
    gamma: float | None = None,
) -> None:
    data = image[CROP] if crop else image
    norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax) if gamma else None
    ax.imshow(data, cmap="gray", vmin=None if norm else 0, vmax=None if norm else vmax,
              norm=norm, interpolation="nearest")
    if title:
        ax.set_title(title, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)


def _load_metrics() -> dict:
    path = RESULTS / "metrics.json"
    if not path.exists():
        raise SystemExit("results/metrics.json is missing: run scripts/evaluate.py first")
    return json.loads(path.read_text(encoding="utf-8"))


_background_cache: dict[int, np.ndarray] = {}


def _background(paths: list[str]) -> np.ndarray:
    """Estimate the background once per frame count (it is reused by 4 figures)."""
    if len(paths) not in _background_cache:
        from photon_clean import estimate_background_from_paths

        _background_cache[len(paths)] = estimate_background_from_paths(paths)[0]
    return _background_cache[len(paths)]


# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #
def figure_hero(metrics: dict) -> None:
    frames = list_images(str(DATA / "frames"))
    background = _background(frames)

    raw = load_gray(frames[0])
    sub = subtract_background(raw, background)
    step1 = remove_hot_pixels(sub)
    step2 = remove_anomalous_clusters(step1)
    clean = median(step2, disk(1)).astype(np.uint8)

    stats = metrics["first_frame_images"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.0))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.855, bottom=0.055, wspace=0.04, hspace=0.30)

    panels = [
        (raw, "1. Raw frame", f"{raw.mean():.2f} ADU mean · every pixel non-zero"),
        (sub, "2. After static background", f"{stats['bg_subtracted']['nonzero_share']:.1%} of pixels left · {stats['bg_subtracted']['events']:,} events"),
        (clean, "3. After cleaning", f"{stats['median']['nonzero_share']:.1%} of pixels left · {stats['median']['events']:,} events"),
    ]
    for column, (image, title, subtitle) in enumerate(panels):
        _frame(axes[0, column], image, title, vmax=60, gamma=0.45)
        axes[0, column].text(
            0.5, -0.03, subtitle, transform=axes[0, column].transAxes,
            ha="center", va="top", fontsize=9, color=MUTED,
        )
        _frame(axes[1, column], image, "", vmax=60, gamma=0.45, crop=True)

    fig.text(
        0.5, 0.985,
        "Photon-counting camera frame: static pedestal, hot pixels, real photons",
        ha="center", va="top", fontsize=16, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.945,
        "1360 x 1040 px, 8-bit, gamma-stretched for display. A 3 ADU sensor bias sits on every pixel "
        "and hides the sparse photon signal.",
        ha="center", va="top", fontsize=9.5, color=MUTED,
    )
    axes[1, 0].set_title("detail · 320 x 244 crop, 1:1 pixels", loc="left", fontsize=9, color=MUTED, pad=6)
    fig.savefig(ASSETS / "hero.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# pipeline diagram
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, title, body, color) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=1.5, edgecolor=color, facecolor=color + "10",
        )
    )
    ax.text(x, y + h * 0.24, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x, y - h * 0.16, body, ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.6)


def _arrow(ax, x0, x1, y) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=MUTED,
            shrinkA=0, shrinkB=0,
        )
    )


def figure_pipeline(metrics: dict) -> None:
    cleaning = metrics["cleaning"]["stages"]
    fig, ax = plt.subplots(figsize=(15.0, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(20, 86)
    ax.axis("off")

    width, gap = 13.0, 3.2
    centers = [3 + width / 2 + i * (width + gap) for i in range(6)]
    boxes = [
        ("Frame stack", "N raw frames\n1360 x 1040 x 8 bit", BLUE),
        ("Background", "per-pixel median\n+ std over the stack", BLUE),
        ("Subtract", "frame - background,\nclipped at 0", BLUE),
        ("Hot pixels", "isolated bright pixel,\nmean(3x3) < 1", ORANGE),
        ("Clusters", "local ratio > 4\nand flat (5x5)", ORANGE),
        ("Median", "disk(1)\nfilter", ORANGE),
    ]
    y_center, height = 62.0, 27.0
    for x, (title, body, color) in zip(centers, boxes):
        _box(ax, x, y_center, width, height, title, body, color)

    for left, right in zip(centers[:-1], centers[1:]):
        _arrow(ax, left + width / 2 + 0.5, right - width / 2 - 0.5, y_center)

    ax.add_patch(
        FancyBboxPatch(
            (centers[3] - width / 2 - 1.6, y_center - height / 2 - 4.0),
            centers[5] - centers[3] + width + 3.2,
            height + 8.0,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=1.1, linestyle=(0, (4, 3)), edgecolor=ORANGE, facecolor="none",
        )
    )
    ax.text(
        (centers[3] + centers[5]) / 2, y_center + height / 2 + 6.0,
        "single-frame cleaning (vectorised)", ha="center", fontsize=9.5, color=ORANGE,
    )

    notes = [
        f"{cleaning['raw']['nonzero_share']:.0%} px",
        "one image,\nreused",
        f"{cleaning['bg_subtracted']['nonzero_share']:.1%} px",
        f"{cleaning['hot_pixels']['nonzero_share']:.1%} px",
        f"{cleaning['clusters']['nonzero_share']:.1%} px",
        f"{cleaning['median']['nonzero_share']:.1%} px",
    ]
    for x, text in zip(centers, notes):
        ax.text(x, 40.0, text, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.4)

    ax.text(
        3.0, 27.0,
        "below each step: share of non-zero pixels in the frame",
        ha="left", va="center", fontsize=8.5, color=MUTED, style="italic",
    )

    ax.set_title("The pipeline", fontsize=16, fontweight="bold", pad=16, color=INK)
    fig.savefig(ASSETS / "pipeline.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# background validation
# --------------------------------------------------------------------------- #
def figure_background(metrics: dict) -> None:
    background_stats = metrics["background"]
    frames = list_images(str(DATA / "frames"))
    background = _background(frames)
    reference = load_gray(str(DATA / "reference_background.png"))
    diff = np.abs(background.astype(np.int16) - reference.astype(np.int16))

    fig = plt.figure(figsize=(13.2, 6.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.34, wspace=0.16)

    for column, (image, title) in enumerate(
        [(background, "Estimated from 24 frames"), (reference, "Dedicated background frame")]
    ):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(image, cmap="gray", vmin=0, vmax=8, interpolation="nearest")
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 2])
    im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=8, interpolation="nearest")
    ax.set_title("|difference|", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    bar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    bar.set_label("ADU", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(grid[1, :2])
    bins = np.arange(0, 12) - 0.5
    ax.hist(background.ravel(), bins=bins, log=True, color=BLUE, alpha=0.75, label="estimated")
    ax.hist(reference.ravel(), bins=bins, log=True, histtype="step", linewidth=1.6, color=ORANGE, label="reference frame")
    ax.set_xlabel("pixel value, ADU")
    ax.set_ylabel("pixels (log scale)")
    ax.set_title("Both images are dominated by a constant 3 ADU pedestal", pad=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        f"MSE        {background_stats['mse']:.3f}\n"
        f"PSNR       {background_stats['psnr_db']:.2f} dB\n"
        f"SSIM       {background_stats['ssim']:.4f}\n"
        f"max |err|  {background_stats['max_abs_error_adu']} ADU\n"
        f"mean |err| {background_stats['mean_abs_error_adu']:.3f} ADU",
        va="top", ha="left", fontsize=9, family="DejaVu Sans Mono",
        linespacing=1.75, color=INK,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f6f8fa", edgecolor=GRID),
    )
    ax.text(
        0.0, 0.34,
        "The estimate is smoother than a single\n"
        "exposure: the median rejects pixels that\n"
        "fire in only part of the stack.\n\n"
        "The faint beam trace appears because only\n"
        "24 frames are shipped; with the full\n"
        "120-frame capture the same code reaches\n"
        "55.79 dB / SSIM 0.9969.",
        va="top", ha="left", fontsize=8.4, linespacing=1.55, color=MUTED,
    )

    fig.suptitle(
        "Background estimate vs an independent background exposure",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.savefig(ASSETS / "background_validation.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# cleaning stages
# --------------------------------------------------------------------------- #
def figure_stages(metrics: dict) -> None:
    frames = list_images(str(DATA / "frames"))
    background = _background(frames)
    sub = subtract_background(load_gray(frames[0]), background)
    step1 = remove_hot_pixels(sub)
    step2 = remove_anomalous_clusters(step1)
    step3 = median(step2, disk(1)).astype(np.uint8)

    removed_hot = (sub > 0) & (step1 == 0)
    removed_cluster = (step1 > 0) & (step2 == 0)
    removed_median = (step2 > 0) & (step3 == 0)

    stages = [
        (sub, "Background subtracted", int((sub > 0).sum()), event_count(sub)),
        (step1, "- hot pixels", int((step1 > 0).sum()), event_count(step1)),
        (step2, "- uniform clusters", int((step2 > 0).sum()), event_count(step2)),
        (step3, "- median filter", int((step3 > 0).sum()), event_count(step3)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.12, wspace=0.06)

    for ax, (image, title, nonzero, events) in zip(axes, stages):
        _frame(ax, image, title, vmax=45)
        ax.text(
            0.5, -0.05, f"{nonzero:,} px  ·  {events:,} events",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color=MUTED,
        )

    base = np.clip(sub.astype(np.float32) / 60.0, 0, 1)
    rgb = np.dstack([base, base, base])
    rgb[removed_hot] = [0.81, 0.13, 0.18]
    rgb[removed_cluster] = [0.95, 0.55, 0.05]
    rgb[removed_median] = [0.10, 0.45, 0.85]
    axes[4].imshow(rgb, interpolation="nearest")
    axes[4].set_title("Removed pixels", pad=8)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    for spine in axes[4].spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    axes[4].legend(
        handles=[
            Patch(facecolor=[0.81, 0.13, 0.18], label=f"hot pixels ({int(removed_hot.sum()):,})"),
            Patch(facecolor=[0.95, 0.55, 0.05], label=f"clusters ({int(removed_cluster.sum()):,})"),
            Patch(facecolor=[0.10, 0.45, 0.85], label=f"median ({int(removed_median.sum()):,})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=8, ncol=1, handlelength=1.2,
    )

    fig.suptitle(
        "Stage by stage on one frame: what each filter removes",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.savefig(ASSETS / "cleaning_stages.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# intensity profiles
# --------------------------------------------------------------------------- #
def figure_profiles(metrics: dict) -> None:
    frames = list_images(str(DATA / "frames"))
    background = _background(frames)
    raw = load_gray(frames[0])
    sub = subtract_background(raw, background)
    clean = median(remove_anomalous_clusters(remove_hot_pixels(sub)), disk(1)).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.2)

    columns = np.arange(raw.shape[1])
    axes[0].plot(columns, raw.mean(axis=0), color=ORANGE, linewidth=1.3, label="raw")
    axes[0].plot(columns, sub.mean(axis=0), color=BLUE, linewidth=1.0, label="background subtracted")
    axes[0].plot(columns, clean.mean(axis=0), color=GREEN, linewidth=1.0, label="cleaned")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("mean intensity, ADU")
    axes[0].set_title("The 3 ADU pedestal is removed from every column", pad=8)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(color=GRID, linewidth=0.6, alpha=0.7)
    axes[0].set_xlim(0, raw.shape[1])

    axes[1].plot(columns, sub.sum(axis=0) / sub.sum(axis=0).max(), color=BLUE, linewidth=1.0, label="background subtracted")
    axes[1].plot(columns, clean.sum(axis=0) / clean.sum(axis=0).max(), color=GREEN, linewidth=1.0, label="cleaned")
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("normalised column energy")
    axes[1].set_title("Same photon structure, less noise between the peaks", pad=8)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(color=GRID, linewidth=0.6, alpha=0.7)
    axes[1].set_xlim(0, raw.shape[1])

    fig.suptitle("Intensity profiles before and after cleaning", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(ASSETS / "intensity_profile.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# downscaling
# --------------------------------------------------------------------------- #
def figure_downscaling(metrics: dict) -> None:
    down = metrics["downscaling"]
    name = down["images"][0]["image"]
    full = load_gray(str(DATA / "full" / "target" / name))
    resized = np.asarray(Image.fromarray(full).resize((512, 512), Image.Resampling.LANCZOS))

    fig = plt.figure(figsize=(13.2, 6.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.34, wspace=0.22)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(full[CROP], cmap="gray", vmin=0, vmax=60, interpolation="nearest")
    ax.set_title("1360 x 1040, cleaned", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 1])
    y0, y1 = CROP[0].start / 2.03125, CROP[0].stop / 2.03125
    x0, x1 = CROP[1].start / 2.65625, CROP[1].stop / 2.65625
    ax.imshow(resized[int(y0):int(y1), int(x0):int(x1)], cmap="gray", vmin=0, vmax=60, interpolation="nearest")
    ax.set_title("after 512 x 512 resize (same field of view)", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 2])
    labels = ["single-pixel\nevents", "2-4 px\nevents", "5+ px\nevents"]
    full_share = [
        np.mean([r["full"]["share_area_1"] for r in down["images"]]),
        np.mean([r["full"]["share_area_2_4"] for r in down["images"]]),
        np.mean([r["full"]["share_area_5_plus"] for r in down["images"]]),
    ]
    small_share = [
        np.mean([r["resized"]["share_area_1"] for r in down["images"]]),
        np.mean([r["resized"]["share_area_2_4"] for r in down["images"]]),
        np.mean([r["resized"]["share_area_5_plus"] for r in down["images"]]),
    ]
    positions = np.arange(3)
    ax.bar(positions - 0.2, full_share, width=0.38, color=BLUE, label="full resolution")
    ax.bar(positions + 0.2, small_share, width=0.38, color=ORANGE, label="512 x 512")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("share of photon events")
    ax.set_ylim(0, 1)
    ax.set_title("Event shape changes", pad=8)
    ax.legend(loc="upper center", fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 0])
    names = [r["image"].replace("_image_", " ").replace(".png", "") for r in down["images"]]
    full_events = [r["full"]["events"] for r in down["images"]]
    small_events = [r["resized"]["events"] for r in down["images"]]
    positions = np.arange(len(names))
    ax.bar(positions - 0.2, full_events, width=0.38, color=BLUE, label="full resolution")
    ax.bar(positions + 0.2, small_events, width=0.38, color=ORANGE, label="512 x 512")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("photon events")
    ax.set_title("Events per frame", pad=8)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 1])
    for image, label, color in [(full, "full resolution", BLUE), (resized, "512 x 512", ORANGE)]:
        values = image[image > 0]
        ax.hist(values, bins=np.arange(0, 260, 8), histtype="step", linewidth=1.6, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value, ADU")
    ax.set_ylabel("pixels (log scale)")
    ax.set_title("Brightness histogram", pad=8)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    summary = (
        f"events kept after resize   {down['mean_event_ratio']:.0%}\n"
        f"mean intensity preserved   {down['mean_intensity_ratio']:.0%}\n"
        f"aspect ratio distorted     {down['aspect_distortion']:.2f}x\n\n"
        "A linear resize keeps the average brightness but\n"
        "merges neighbouring photons into one blob and\n"
        f"erases single-pixel events "
        f"({down['mean_single_pixel_share_full']:.0%} -> "
        f"{down['mean_single_pixel_share_resized']:.0%})."
    )
    ax.text(
        0.0, 0.95, summary, va="top", ha="left", fontsize=10, family="DejaVu Sans Mono",
        linespacing=1.6, color=INK,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f6f8fa", edgecolor=GRID),
    )

    fig.suptitle(
        "What the 512 x 512 preprocessing did to the photon signal",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.savefig(ASSETS / "downscaling.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# U-Net failure
# --------------------------------------------------------------------------- #
def figure_unet(metrics: dict) -> None:
    unet = metrics["unet"]
    arrays = {}
    for kind in ("input", "predicted", "true"):
        arrays[kind] = np.asarray(
            Image.open(LEGACY / "unet" / f"test_001_{kind}.png").convert("L"), dtype=np.uint8
        )

    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.30)

    titles = {
        "input": ("Network input (512 x 512)", f"mean {arrays['input'].mean():.2f} ADU"),
        "predicted": ("U-Net prediction", f"mean {arrays['predicted'].mean():.2f} ADU"),
        "true": ("Target (classical pipeline)", f"mean {arrays['true'].mean():.2f} ADU"),
    }
    for column, kind in enumerate(("input", "predicted", "true")):
        ax = fig.add_subplot(grid[0, column])
        _frame(ax, arrays[kind], titles[kind][0], vmax=30)
        ax.text(
            0.5, -0.04, titles[kind][1], transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color=MUTED,
        )

    ax = fig.add_subplot(grid[1, 0])
    for kind, color in (("input", BLUE), ("predicted", ORANGE), ("true", GREEN)):
        ax.hist(arrays[kind].ravel(), bins=np.arange(0, 40), histtype="step", linewidth=1.6, color=color, label=kind)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value, ADU")
    ax.set_ylabel("pixels (log scale)")
    ax.set_title("Prediction histogram", pad=8)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 1])
    samples = unet["samples"]
    positions = np.arange(len(samples))
    ax.bar(positions - 0.26, [s["input_mean"] for s in samples], width=0.24, color=BLUE, label="input")
    ax.bar(positions, [s["predicted_mean"] for s in samples], width=0.24, color=ORANGE, label="prediction")
    ax.bar(positions + 0.26, [s["target_mean"] for s in samples], width=0.24, color=GREEN, label="target")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"sample {s['sample']}" for s in samples], fontsize=8.5)
    ax.set_ylabel("mean intensity, ADU")
    ax.set_ylim(0, 12.5)
    ax.set_title("Mean intensity per sample", pad=8)
    ax.legend(fontsize=8.5, loc="upper center", ncol=3, handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = fig.add_subplot(grid[1, 2])
    diff = np.abs(arrays["predicted"].astype(np.int16) - arrays["true"].astype(np.int16))
    im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=60, interpolation="nearest")
    ax.set_title("Error map", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("ADU", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    ax.text(
        0.5, -0.04, f"energy ratio {unet['mean_energy_ratio']:.0f}x the target",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=MUTED,
    )

    fig.suptitle(
        "The U-Net baseline: a smooth grey haze instead of photons",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.savefig(ASSETS / "unet_failure.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def figure_benchmark() -> None:
    path = RESULTS / "benchmark.json"
    if not path.exists():
        print("results/benchmark.json missing, skipping benchmark figure")
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    labels = ["Python loop\n(reference)", "Vectorised\nhot pixels", "Vectorised\nfull chain"]
    values = [data["loop_seconds"], data["vectorised_seconds"], data["full_pipeline_seconds"]]
    colors = [MUTED, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    positions = np.arange(len(labels))
    bars = ax.barh(positions, values, color=colors, height=0.45)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per 1360 x 1040 frame (log scale)")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_xlim(min(values) / 2, max(values) * 2.2)

    for bar, value in zip(bars, values):
        ax.text(
            value * 1.15, bar.get_y() + bar.get_height() / 2,
            f"{value * 1000:.1f} ms  ·  {1 / value:.0f} fps",
            va="center", fontsize=9, color=INK,
        )

    ax.set_title(
        f"Bit-identical output, {data['speedup']:.0f}x faster than the reference loop",
        pad=12,
    )
    fig.savefig(ASSETS / "benchmark.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def figure_social_preview(metrics: dict) -> None:
    """1280x640 card for GitHub Settings -> Social preview (not used in the README)."""
    frames = list_images(str(DATA / "frames"))
    background = _background(frames)
    raw = load_gray(frames[0])
    sub = subtract_background(raw, background)
    clean = median(remove_anomalous_clusters(remove_hot_pixels(sub)), disk(1)).astype(np.uint8)

    speedup = "37x"
    bench_path = RESULTS / "benchmark.json"
    if bench_path.exists():
        speedup = f"{json.loads(bench_path.read_text(encoding='utf-8'))['speedup']:.0f}x"

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("white")
    fig.add_artist(
        plt.Rectangle((0, 0), 0.012, 1, transform=fig.transFigure, facecolor=BLUE, edgecolor="none")
    )

    fig.text(0.055, 0.885, "Photon frame cleaning", fontsize=30, fontweight="bold", color=INK, va="top")
    fig.text(
        0.055, 0.775,
        "Removing a static sensor pedestal and hot pixels\nfrom 8-bit photon-counting camera frames",
        fontsize=14.5, color=MUTED, va="top", linespacing=1.55,
    )

    stats = [
        (f"{metrics['background']['psnr_db']:.1f} dB", "background estimate vs an\nindependent exposure"),
        ("100 %", "of the energy above 50 ADU\nis preserved"),
        (speedup, "faster than the per-pixel\nreference loop"),
    ]
    y = 0.545
    for value, label in stats:
        fig.text(0.055, y, value, fontsize=24, fontweight="bold", color=BLUE, va="center")
        fig.text(0.235, y, label, fontsize=10.5, color=MUTED, va="center", linespacing=1.45)
        y -= 0.135

    fig.text(
        0.055, 0.075, "github.com/d1m40fficial/photon-denoising",
        fontsize=10.5, color=MUTED, va="center",
    )

    crop = (slice(400, 600), slice(480, 880))
    panels = [(raw, "raw frame  ·  3 ADU pedestal"), (clean, "after cleaning  ·  310 events")]
    for index, (image, label) in enumerate(panels):
        y0 = 0.48 - index * 0.46
        ax = fig.add_axes([0.52, y0, 0.42, 0.42])
        ax.imshow(
            image[crop], cmap="gray", interpolation="nearest",
            norm=matplotlib.colors.PowerNorm(gamma=0.45, vmin=0, vmax=60),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.8)
        ax.text(
            0.022, 0.95, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#00000088", edgecolor="none"),
        )

    fig.savefig(ASSETS / "social_preview.png", dpi=100, facecolor="white")
    plt.close(fig)


def main() -> int:
    style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()

    print("hero ...")
    figure_hero(metrics)
    print("pipeline ...")
    figure_pipeline(metrics)
    print("background ...")
    figure_background(metrics)
    print("stages ...")
    figure_stages(metrics)
    print("profiles ...")
    figure_profiles(metrics)
    print("downscaling ...")
    figure_downscaling(metrics)
    print("unet ...")
    figure_unet(metrics)
    print("benchmark ...")
    figure_benchmark()
    print("social preview ...")
    figure_social_preview(metrics)

    for file in sorted(ASSETS.glob("*.png")):
        print(f"  {file.name:<28}{file.stat().st_size / 1024:7.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
