"""Benchmark: literal per-pixel Python loop vs the vectorised implementation.

The reference implementation below is the rule documented in
:func:`photon_clean.cleaning.remove_hot_pixels`, written out one pixel at a time
so that the two can be compared for both speed *and* equality.

    python scripts/benchmark.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from _bootstrap import DATA, RESULTS

from photon_clean import (
    clean_image,
    estimate_background_from_paths,
    list_images,
    load_gray,
    remove_hot_pixels,
    subtract_background,
)


def loop_remove_hot_pixels(
    image: np.ndarray,
    intensity_threshold: int = 1,
    neighbour_mean: float = 1.0,
    radius: int = 1,
) -> np.ndarray:
    """Reference implementation: one Python iteration per pixel."""
    image = np.asarray(image, dtype=np.uint8)
    cleaned = image.copy()
    height, width = image.shape
    padded = np.pad(image, radius, mode="reflect")

    for y in range(height):
        for x in range(width):
            value = padded[y + radius, x + radius]
            if value <= intensity_threshold:
                continue
            window = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            mask = window > 0
            mask[radius, radius] = False
            mean_neighbours = window[mask].mean() if mask.any() else 0.0
            if mean_neighbours < neighbour_mean and value > 3.0 * mean_neighbours:
                cleaned[y, x] = 0
    return cleaned


def timed(func, *args, repeats: int, **kwargs) -> float:
    """Best-of-N wall-clock time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best, result


def timed_pair(first, second, *args, repeats: int):
    """Time two functions alternately, so both see the same machine load."""
    best_first = best_second = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result_first = first(*args)
        best_first = min(best_first, time.perf_counter() - start)

        start = time.perf_counter()
        result_second = second(*args)
        best_second = min(best_second, time.perf_counter() - start)
    return best_first, result_first, best_second, result_second


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    frames = list_images(str(DATA / "frames"))
    background, _ = estimate_background_from_paths(frames)
    raw = load_gray(frames[0])
    frame = subtract_background(raw, background)
    pixels = frame.size

    print(f"frame: {frame.shape[1]}x{frame.shape[0]} ({pixels:,} px), {args.repeats} repeats")

    loop_time, loop_result, vec_time, vec_result = timed_pair(
        loop_remove_hot_pixels, remove_hot_pixels, frame, repeats=args.repeats
    )
    pipeline_time, _ = timed(clean_image, frame, repeats=args.repeats)

    identical = bool(np.array_equal(loop_result, vec_result))
    differing = int((loop_result != vec_result).sum())

    results = {
        "frame_shape": list(frame.shape),
        "pixels": int(pixels),
        "repeats": args.repeats,
        "loop_seconds": loop_time,
        "vectorised_seconds": vec_time,
        "full_pipeline_seconds": pipeline_time,
        "speedup": loop_time / vec_time,
        "identical": identical,
        "differing_pixels": differing,
        "loop_fps": 1.0 / loop_time,
        "vectorised_fps": 1.0 / vec_time,
        "pipeline_fps": 1.0 / pipeline_time,
        "throughput_mpx_per_s": pixels / vec_time / 1e6,
        "original_notebook_fps": 1.2,
    }

    print(f"{'implementation':<24}{'s / frame':>12}{'frames / s':>14}")
    print(f"{'Python loop':<24}{loop_time:>12.3f}{1.0 / loop_time:>14.2f}")
    print(f"{'vectorised (hot pixels)':<24}{vec_time:>12.4f}{1.0 / vec_time:>14.1f}")
    print(f"{'vectorised (full chain)':<24}{pipeline_time:>12.4f}{1.0 / pipeline_time:>14.1f}")
    print(f"\nspeed-up: {results['speedup']:.0f}x, identical output: {identical} ({differing} differing pixels)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
