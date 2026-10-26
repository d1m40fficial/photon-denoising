# How the pipeline works

## The problem

A photon-counting camera returns 8-bit frames of 1360 x 1040 pixels. Three
different things end up in those numbers:

1. **A static pedestal.** Every pixel gets the same additive offset from the
   sensor electronics. On this camera it is ~3 ADU, so ~98.6 % of the pixels of a
   "background" frame have exactly the value 3 and the whole frame looks grey.
2. **Sensor defects.** Hot pixels fire at a fixed location and look like small
   bright dots. They can be single pixels or small crosses of 2–6 pixels.
3. **Real photons.** A photon produces a diffraction-limited spot of a few pixels
   with intensity from 10 to 255 ADU.

The goal is to keep (3) and remove (1) and (2).

## Step 1 — estimate the static background

Photons land on random pixels, so the *median* of a stack of frames is a robust
estimate of the fixed pattern: at a pixel that only occasionally receives a
photon the median is the pedestal, not the photon.

For a stack of `N` frames and a batch size `B`:

```
for each batch of B frames:
    median_map = median(frames, axis=0)
    std_map    = std(frames, axis=0)
    stable     = (std_map < 50) and (median_map > 1)

background = mean(median_map[stable])   # averaged over all batches
```

Two conditions matter:

* `std < 50` throws away pixels that flicker too much between frames — either
  they are hit by photons often or they are genuinely unstable. Without this
  filter a few hot pixels would be baked into the background and then subtracted
  from every frame.
* `median > 1` keeps the pedestal and ignores pixels that are always black.

Batching means a stack of thousands of frames never has to be held in memory: a
batch of 200 frames at 1360 x 1040 x `float32` is ~1.1 GB.

## Step 2 — subtract

```
frame_clean = clip(frame - background, 0, 255)
```

The background is zero where no stable pedestal was found, so those pixels pass
through unchanged. On this sensor the subtraction turns a frame in which *every*
pixel was non-zero into one where only ~5.9 % of the pixels are.

## Step 3 — remove isolated hot pixels

A pixel is deleted when all of the following hold:

* it is brighter than 1 ADU;
* the mean of its non-zero neighbours in a 3x3 window is below 1 ADU;
* it is more than 3x brighter than that mean.

The intuition is that an isolated hot pixel has nothing around it, while a
photon spot always has at least a couple of neighbouring pixels above zero.

## Step 4 — remove uniform clusters

Some defects span several pixels and survive step 3. For those:

* compute the local mean over a 3x3 window;
* flag pixels whose ratio to the local mean is above 4 (or below 1/4);
* keep only the flagged pixels that are *flat*, i.e. within 15 % of a Gaussian
  smoothed version of the image.

Photons are sharp peaks, so the flatness test protects them: a real photon is far
from the smoothed image, a defect plateau is not.

## Step 5 — median filter

A `disk(1)` median filter removes the remaining isolated residues and evens out
the edges of the detected spots. It is the last step and the cheapest one.

## Why every step is vectorised

The original implementation looped over 1.4 million pixels in Python for step 3.
The current version builds the same decision with two `cv2.filter2D` calls:

```
neighbour_sum   = filter2D(image, ones(3x3) - centre)
neighbour_count = filter2D(image > 0, ones(3x3) - centre)
neighbour_mean  = neighbour_sum / neighbour_count      (0 where count == 0)
hot = (image > 1) & (neighbour_mean < 1) & (image > 3 * neighbour_mean)
```

`scripts/benchmark.py` checks that the vectorised output is **bit-identical** to
the per-pixel reference and measures the speed-up (35–40x on a Ryzen 7 / Python
3.14, depending on machine load).

## Parameters

All of them live in `photon_clean.CleanConfig`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `hot_pixel_intensity` | 1 | minimum brightness to be tested as a defect |
| `hot_pixel_neighbour_mean` | 1.0 | neighbour mean below which a pixel counts as isolated |
| `hot_pixel_radius` | 1 | 3x3 neighbourhood |
| `cluster_intensity_ratio` | 4.0 | local contrast that makes a cluster anomalous |
| `cluster_similarity` | 0.15 | flatness tolerance (fraction of full scale) |
| `cluster_radius` | 1 | 3x3 local mean |
| `median_radius` | 1 | `disk(1)` median filter, 0 disables it |

They were tuned by hand on the full dataset; `docs/results.md` shows what each
step removes and what it costs.

## Known limitations

* **Single-pixel photons are indistinguishable from hot pixels.** The filter
  uses only local contrast, so a real photon that landed alone on the sensor is
  removed together with the defect. Measured cost: 0 % of the energy above
  50 ADU is lost, but 23 % of the total energy (which lives in the 1–5 ADU noise
  floor) disappears.
* **A static defect with a very stable intensity is absorbed into the
  background** instead of being cleaned, because the median sees it as part of
  the fixed pattern.
* **The background must be estimated from the same capture.** A background from
  another session or another camera does not transfer.
* **The method is per-frame and stateless.** It cannot use the fact that a
  photon is a one-off event while a defect recurs at the same coordinate — see
  the roadmap in the main README.
