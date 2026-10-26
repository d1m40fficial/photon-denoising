"""Re-measure every number quoted in the README and write ``results/metrics.json``.

Sections
--------
background    estimated static background vs the dedicated reference frame
stages        what each cleaning step removes (pixels, events, energy)
persistence   how many deleted pixels recur across frames (i.e. are real defects)
fidelity      this implementation vs the outputs of the original notebook
downscaling   what the 1360x1040 -> 512x512 resize did to the photon signal
unet          statistics of the legacy U-Net predictions

Usage
-----
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk

from _bootstrap import DATA, LEGACY, RESULTS

from photon_clean import (
    CleanConfig,
    clean_image,
    estimate_background_from_paths,
    event_count,
    list_images,
    load_gray,
    mse,
    photon_intensity_ratio,
    psnr,
    remove_anomalous_clusters,
    remove_hot_pixels,
    ssim,
    subtract_background,
)

THRESHOLDS = (1, 2, 3, 5, 10, 20, 50, 100)


def _num(value):
    """Convert numpy scalars to plain Python numbers for JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _component_areas(image: np.ndarray) -> np.ndarray:
    import cv2

    binary = (np.asarray(image) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.empty(0, dtype=np.int32)
    return stats[1:, cv2.CC_STAT_AREA]


def _area_profile(areas: np.ndarray) -> dict:
    if areas.size == 0:
        return {"events": 0, "share_area_1": 0.0, "share_area_2_4": 0.0, "share_area_5_plus": 0.0}
    return {
        "events": int(areas.size),
        "share_area_1": float((areas == 1).mean()),
        "share_area_2_4": float(((areas >= 2) & (areas <= 4)).mean()),
        "share_area_5_plus": float((areas >= 5).mean()),
        "mean_area": float(areas.mean()),
    }


# --------------------------------------------------------------------------- #
# 1. background
# --------------------------------------------------------------------------- #
def measure_background(frames: list[str]) -> dict:
    background, std_map = estimate_background_from_paths(frames)
    reference = load_gray(str(DATA / "reference_background.png"))

    diff = background.astype(np.int16) - reference.astype(np.int16)
    values, counts = np.unique(background, return_counts=True)
    mode = int(values[np.argmax(counts)])

    return {
        "frames_used": len(frames),
        "shape": list(background.shape),
        "mean_adu": float(background.mean()),
        "mode_adu": mode,
        "mode_share": float(counts.max() / background.size),
        "nonzero_share": float((background > 0).mean()),
        "reference_mean_adu": float(reference.mean()),
        "mse": mse(reference, background),
        "psnr_db": psnr(reference, background),
        "ssim": ssim(reference, background),
        "max_abs_error_adu": int(np.abs(diff).max()),
        "mean_abs_error_adu": float(np.abs(diff).mean()),
        "std_map_median": float(np.median(std_map)),
    }


# --------------------------------------------------------------------------- #
# 2. cleaning stages
# --------------------------------------------------------------------------- #
def measure_stages(frames: list[str], background: np.ndarray) -> dict:
    cfg = CleanConfig()
    totals = {name: {"energy": 0.0, "nonzero": 0, "events": 0} for name in
              ("raw", "bg_subtracted", "hot_pixels", "clusters", "median")}
    bucket_energy = {t: {"total": 0.0, "kept": 0.0} for t in THRESHOLDS}
    bright_total = bright_kept = 0.0

    first_frame = None
    for path in frames:
        raw = load_gray(path)
        sub = subtract_background(raw, background)
        step1 = remove_hot_pixels(
            sub,
            intensity_threshold=cfg.hot_pixel_intensity,
            neighbour_mean=cfg.hot_pixel_neighbour_mean,
            radius=cfg.hot_pixel_radius,
        )
        step2 = remove_anomalous_clusters(
            step1,
            intensity_ratio=cfg.cluster_intensity_ratio,
            similarity_threshold=cfg.cluster_similarity,
            radius=cfg.cluster_radius,
        )
        step3 = median(step2, disk(cfg.median_radius)).astype(np.uint8)

        for name, image in zip(
            ("raw", "bg_subtracted", "hot_pixels", "clusters", "median"),
            (raw, sub, step1, step2, step3),
        ):
            totals[name]["energy"] += float(image.sum())
            totals[name]["nonzero"] += int((image > 0).sum())
            totals[name]["events"] += event_count(image)

        kept = step3 > 0
        for t in THRESHOLDS:
            mask = sub > t
            bucket_energy[t]["total"] += float(sub[mask].sum())
            bucket_energy[t]["kept"] += float(sub[mask & kept].sum())

        bright = sub > 50
        bright_total += float(sub[bright].sum())
        bright_kept += float(sub[bright & kept].sum())

        if first_frame is None:
            first_frame = {
                "raw": raw,
                "bg_subtracted": sub,
                "hot_pixels": step1,
                "clusters": step2,
                "median": step3,
            }

    n = len(frames)
    pixels = first_frame["raw"].size
    for name, data in totals.items():
        data["energy_per_frame"] = data["energy"] / n
        data["nonzero_share"] = data["nonzero"] / (n * pixels)
        data["events_per_frame"] = data["events"] / n

    retention = {
        str(t): {
            "energy_share_kept": bucket_energy[t]["kept"] / max(bucket_energy[t]["total"], 1e-9),
            "energy_total": bucket_energy[t]["total"],
            "energy_kept": bucket_energy[t]["kept"],
        }
        for t in THRESHOLDS
    }

    return {
        "stages": totals,
        "energy_retention_by_intensity": retention,
        "bright_signal_retention": bright_kept / max(bright_total, 1e-9),
        "first_frame": {k: v for k, v in first_frame.items()},
    }


# --------------------------------------------------------------------------- #
# 3. persistence of deleted pixels
# --------------------------------------------------------------------------- #
def measure_persistence(frames: list[str], background: np.ndarray) -> dict:
    stack = np.stack([subtract_background(load_gray(p), background) for p in frames])
    hits = (stack > 0).sum(axis=0)

    removed = np.zeros(stack.shape[1:], dtype=np.int32)
    for frame in stack:
        out = remove_hot_pixels(frame)
        removed += ((frame > 0) & (out == 0)).astype(np.int32)

    removed_mask = removed > 0
    total_removed = int(removed_mask.sum())
    recurring = int((removed[removed_mask] >= 6).sum())

    return {
        "frames": int(stack.shape[0]),
        "pixels_hit_once": int((hits == 1).sum()),
        "pixels_hit_6_plus": int((hits >= 6).sum()),
        "pixels_hit_all_frames": int((hits == stack.shape[0]).sum()),
        "removed_pixels_union": total_removed,
        "removed_pixels_recurring_6_plus": recurring,
        "recurring_share": recurring / max(total_removed, 1),
    }


# --------------------------------------------------------------------------- #
# 4. fidelity against the original notebook output
# --------------------------------------------------------------------------- #
def measure_fidelity() -> dict:
    folder = DATA / "full"
    rows = []
    for path in list_images(str(folder / "bg_subtracted")):
        name = Path(path).name
        bg_sub = load_gray(path)
        target = load_gray(str(folder / "target" / name))
        ours = clean_image(bg_sub)
        rows.append(
            {
                "image": name,
                "pixels_exact_share": float((ours == target).mean()),
                "mse": mse(target, ours),
                "psnr_db": psnr(target, ours),
                "ssim": ssim(target, ours),
                "energy_ratio": photon_intensity_ratio(ours, target),
                "events_ours": event_count(ours),
                "events_reference": event_count(target),
            }
        )
    return {
        "images": rows,
        "mean_pixels_exact_share": float(np.mean([r["pixels_exact_share"] for r in rows])),
        "min_psnr_db": float(np.min([r["psnr_db"] for r in rows])),
    }


# --------------------------------------------------------------------------- #
# 5. what the 2.65x downscale did
# --------------------------------------------------------------------------- #
def measure_downscaling() -> dict:
    folder = DATA / "full"
    rows = []
    for path in list_images(str(folder / "target")):
        name = Path(path).name
        full = load_gray(path)
        small_path = DATA / "pairs" / "target" / name
        small = load_gray(str(small_path)) if small_path.exists() else None
        resized = np.asarray(
            Image.fromarray(full).resize((512, 512), Image.Resampling.LANCZOS)
        )

        full_areas = _component_areas(full)
        resized_areas = _component_areas(resized)

        row = {
            "image": name,
            "scale_x": full.shape[1] / 512.0,
            "scale_y": full.shape[0] / 512.0,
            "full": {
                **_area_profile(full_areas),
                "mean_adu": float(full.mean()),
                "peak_adu": int(full.max()),
            },
            "resized": {
                **_area_profile(resized_areas),
                "mean_adu": float(resized.mean()),
                "peak_adu": int(resized.max()),
            },
            # A linear resize preserves the mean intensity but not the number of
            # distinguishable events: neighbouring photons end up in one blob.
            "mean_ratio_resized": float(resized.mean()) / max(float(full.mean()), 1e-9),
        }
        if small is not None:
            row["shipped_512"] = {
                **_area_profile(_component_areas(small)),
                "mean_adu": float(small.mean()),
                "peak_adu": int(small.max()),
            }
        rows.append(row)

    return {
        "images": rows,
        "aspect_distortion": (1360 / 1040) / (512 / 512),
        "mean_event_ratio": float(
            np.mean([r["resized"]["events"] / max(r["full"]["events"], 1) for r in rows])
        ),
        "mean_single_pixel_share_full": float(np.mean([r["full"]["share_area_1"] for r in rows])),
        "mean_single_pixel_share_resized": float(
            np.mean([r["resized"]["share_area_1"] for r in rows])
        ),
        "mean_intensity_ratio": float(np.mean([r["mean_ratio_resized"] for r in rows])),
    }


# --------------------------------------------------------------------------- #
# 6. legacy U-Net predictions
# --------------------------------------------------------------------------- #
def measure_unet() -> dict:
    folder = LEGACY / "unet"
    rows = []
    for index in (0, 1, 2):
        arrays = {
            kind: np.asarray(
                Image.open(folder / f"test_{index:03d}_{kind}.png").convert("L"), dtype=np.uint8
            )
            for kind in ("input", "predicted", "true")
        }
        rows.append(
            {
                "sample": index,
                "input_mean": float(arrays["input"].mean()),
                "predicted_mean": float(arrays["predicted"].mean()),
                "target_mean": float(arrays["true"].mean()),
                "predicted_energy_ratio": photon_intensity_ratio(
                    arrays["predicted"], arrays["true"]
                ),
                "predicted_mse": mse(arrays["true"], arrays["predicted"]),
                "predicted_ssim": ssim(arrays["true"], arrays["predicted"]),
                "predicted_nonzero_share": float((arrays["predicted"] > 0).mean()),
                "target_nonzero_share": float((arrays["true"] > 0).mean()),
            }
        )
    return {
        "samples": rows,
        "mean_predicted_mean": float(np.mean([r["predicted_mean"] for r in rows])),
        "mean_target_mean": float(np.mean([r["target_mean"] for r in rows])),
        "mean_energy_ratio": float(np.mean([r["predicted_energy_ratio"] for r in rows])),
    }


def main() -> int:
    frames = list_images(str(DATA / "frames"))
    if not frames:
        raise SystemExit(f"No sample frames found in {DATA / 'frames'}")

    print(f"frames: {len(frames)} x {load_gray(frames[0]).shape}")
    background, _ = estimate_background_from_paths(frames)

    print("[1/6] background estimate vs reference frame")
    background_stats = measure_background(frames)
    print(f"      PSNR {background_stats['psnr_db']:.2f} dB, SSIM {background_stats['ssim']:.4f}")

    print("[2/6] cleaning stages")
    stage_stats = measure_stages(frames, background)
    first = stage_stats.pop("first_frame")
    print(
        "      nonzero pixels: raw {:.2%} -> bg-sub {:.2%} -> clean {:.2%}".format(
            stage_stats["stages"]["raw"]["nonzero_share"],
            stage_stats["stages"]["bg_subtracted"]["nonzero_share"],
            stage_stats["stages"]["median"]["nonzero_share"],
        )
    )

    print("[3/6] persistence of deleted pixels")
    persistence = measure_persistence(frames, background)
    print(f"      {persistence['recurring_share']:.1%} of deleted pixels recur in 6+ frames")

    print("[4/6] fidelity against the original notebook output")
    fidelity = measure_fidelity()
    print(f"      {fidelity['mean_pixels_exact_share']:.4%} identical pixels")

    print("[5/6] downscaling analysis")
    downscaling = measure_downscaling()
    print(f"      events kept after resize: {downscaling['mean_event_ratio']:.1%}")

    print("[6/6] legacy U-Net predictions")
    unet = measure_unet()
    print(f"      predicted mean {unet['mean_predicted_mean']:.2f} vs target {unet['mean_target_mean']:.2f}")

    payload = {
        "sample_frames": len(frames),
        "frame_shape": list(load_gray(frames[0]).shape),
        "background": background_stats,
        "cleaning": stage_stats,
        "persistence": persistence,
        "fidelity": fidelity,
        "downscaling": downscaling,
        "unet": unet,
        "first_frame_images": {
            key: {
                "mean": float(value.mean()),
                "nonzero_share": float((value > 0).mean()),
                "events": event_count(value),
                "energy": int(value.sum()),
            }
            for key, value in first.items()
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=_num), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
