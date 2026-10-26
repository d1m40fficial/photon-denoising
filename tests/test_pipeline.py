"""Unit tests for the photon_clean pipeline.

Run from the repository root::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_clean import (  # noqa: E402
    CleanConfig,
    clean_image,
    estimate_background,
    event_count,
    mse,
    photon_intensity_ratio,
    remove_hot_pixels,
    subtract_background,
)
from photon_clean.background import background_mask  # noqa: E402


def loop_remove_hot_pixels(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Reference per-pixel implementation used to check the vectorised one."""
    image = np.asarray(image, dtype=np.uint8)
    cleaned = image.copy()
    height, width = image.shape
    padded = np.pad(image, radius, mode="reflect")
    for y in range(height):
        for x in range(width):
            value = padded[y + radius, x + radius]
            if value <= 1:
                continue
            window = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            mask = window > 0
            mask[radius, radius] = False
            mean_neighbours = window[mask].mean() if mask.any() else 0.0
            if mean_neighbours < 1.0 and value > 3.0 * mean_neighbours:
                cleaned[y, x] = 0
    return cleaned


class TestHotPixels(unittest.TestCase):
    def test_matches_reference_loop(self) -> None:
        rng = np.random.default_rng(42)
        image = np.zeros((32, 32), dtype=np.uint8)
        image[5, 5] = 200  # isolated -> removed
        image[10:13, 10:13] = 120  # compact blob -> kept
        image[rng.integers(0, 32, 40), rng.integers(0, 32, 40)] = rng.integers(2, 60, 40)

        expected = loop_remove_hot_pixels(image)
        actual = remove_hot_pixels(image)
        np.testing.assert_array_equal(actual, expected)

    def test_isolated_pixel_is_removed(self) -> None:
        image = np.zeros((9, 9), dtype=np.uint8)
        image[4, 4] = 90
        self.assertEqual(remove_hot_pixels(image)[4, 4], 0)

    def test_large_blob_is_kept(self) -> None:
        image = np.zeros((11, 11), dtype=np.uint8)
        image[4:7, 4:7] = 150
        cleaned = remove_hot_pixels(image)
        self.assertEqual(cleaned[5, 5], 150)
        self.assertGreater(int((cleaned > 0).sum()), 0)


class TestBackground(unittest.TestCase):
    def test_recovers_constant_pedestal(self) -> None:
        rng = np.random.default_rng(0)
        frames = [np.full((16, 16), 7, dtype=np.uint8) for _ in range(5)]
        background, std_map = estimate_background(frames)
        np.testing.assert_array_equal(background, np.full((16, 16), 7, dtype=np.uint8))
        self.assertLess(float(std_map.max()), 1e-6)

    def test_rejects_unstable_pixels(self) -> None:
        frames = [np.full((8, 8), 5, dtype=np.uint8) for _ in range(4)]
        for index, frame in enumerate(frames):
            frame[0, 0] = 10 if index % 2 == 0 else 200  # std ~95 ADU
        background, std_map = estimate_background(frames, std_threshold=50)
        self.assertEqual(int(background[0, 0]), 0)
        self.assertGreater(float(std_map[0, 0]), 50)

    def test_ignores_dark_pixels(self) -> None:
        frames = [np.zeros((4, 4), dtype=np.uint8) for _ in range(3)]
        background, _ = estimate_background(frames)
        self.assertEqual(int(background.max()), 0)
        self.assertFalse(background_mask(background).any())


class TestSubtraction(unittest.TestCase):
    def test_clips_negative_values(self) -> None:
        image = np.array([[1, 5], [9, 12]], dtype=np.uint8)
        background = np.array([[3, 3], [3, 3]], dtype=np.uint8)
        result = subtract_background(image, background)
        np.testing.assert_array_equal(result, np.array([[0, 2], [6, 9]], dtype=np.uint8))

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            subtract_background(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))


class TestCleaningChain(unittest.TestCase):
    def test_single_pixel_noise_is_gone(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        image[3, 3] = 250
        image[20:23, 20:23] = 180
        cleaned = clean_image(image, CleanConfig())
        self.assertEqual(int(cleaned[3, 3]), 0)
        self.assertGreater(int((cleaned > 0).sum()), 0)

    def test_output_is_uint8_and_bounded(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        image[8, 8] = 255
        cleaned = clean_image(image)
        self.assertEqual(cleaned.dtype, np.uint8)
        self.assertLessEqual(int(cleaned.max()), 255)


class TestMetrics(unittest.TestCase):
    def test_identical_images(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2, 2] = 40
        self.assertEqual(mse(image, image), 0.0)
        self.assertAlmostEqual(photon_intensity_ratio(image, image), 1.0)

    def test_event_count(self) -> None:
        image = np.zeros((10, 10), dtype=np.uint8)
        image[1, 1] = 30
        image[5:7, 5:7] = 30
        self.assertEqual(event_count(image), 2)

    def test_intensity_ratio_scales(self) -> None:
        prediction = np.full((4, 4), 20, dtype=np.uint8)
        target = np.full((4, 4), 10, dtype=np.uint8)
        self.assertAlmostEqual(photon_intensity_ratio(prediction, target), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
