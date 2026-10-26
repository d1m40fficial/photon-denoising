# Measured results

Every number below is produced by the scripts in this repository and stored in
`results/metrics.json` and `results/benchmark.json`:

```bash
python scripts/evaluate.py                 # results/metrics.json
python scripts/benchmark.py --repeats 3    # results/benchmark.json
python scripts/make_figures.py             # assets/*.png
```

The sample stack is 24 frames of `cam1` at 1360 x 1040 (`data/samples/frames`).
Results for the full 120-frame capture are quoted separately where they differ.

## 1. Background estimate

The estimate is compared with `data/samples/reference_background.png`, a
dedicated background exposure that the algorithm never sees.

| Metric | 24 frames (shipped) | 120 frames (full capture) |
| --- | --- | --- |
| MSE | 0.200 | 0.171 |
| PSNR | 55.12 dB | 55.79 dB |
| SSIM | 0.9965 | 0.9969 |
| Mean absolute error | 0.084 ADU | 0.076 ADU |
| Maximum absolute error | 44 ADU | — |

The 120-frame column reproduces the figure quoted in the original practice
report (55.78 dB / 0.9969), which is a good sign that the reimplementation is
faithful.

The distribution of both images is a spike at 3 ADU that covers 98.6 % of the
sensor, plus a long tail of defect pixels. The estimate is *smoother* than a
single exposure because the median rejects pixels that fire in only part of the
stack.

## 2. What each cleaning step removes

Averages over the 24-frame stack:

| Stage | Non-zero pixels | Photon events | Mean energy per frame |
| --- | --- | --- | --- |
| Raw frame | 100.00 % | — | 4 665 716 |
| After background subtraction | 5.88 % | 26 034 | 376 516 |
| After hot-pixel filter | 4.59 % | 7 778 | 328 829 |
| After cluster filter | 1.78 % | 1 962 | 274 770 |
| After median filter | 1.16 % | 310 | 250 434 |

"Photon event" = 8-connected component of pixels above zero. The raw frame is a
single component because the pedestal is everywhere non-zero, so that cell is not
meaningful.

### What the cleaning costs

Energy retention as a function of pixel intensity, aggregated over 24 frames:

| Only pixels brighter than | Energy kept |
| --- | --- |
| 1 ADU | 77.0 % |
| 2 ADU | 85.8 % |
| 3 ADU | 92.6 % |
| 5 ADU | 98.4 % |
| 10 ADU | 99.94 % |
| 20 ADU | 99.99 % |
| 50 ADU | 100.00 % |

Total energy falls to **66.5 %**, but the loss is confined to the 1–5 ADU noise
floor: no pixel above 50 ADU is touched. That is the honest version of the
"photons are not suppressed" claim — the pipeline is lossless for the signal and
lossy for the noise, and the two overlap only in the dimmest bin.

### Do the deleted pixels look like defects?

For every pixel removed by the hot-pixel filter we counted in how many of the 24
frames it was removed:

| Deleted pixels | Count | Share |
| --- | --- | --- |
| Deleted at least once | 280 117 | 100 % |
| Deleted in 6 or more frames | 6 990 | 2.5 % |

Only one deleted pixel in forty recurs often enough to be called a permanent
defect. The rest are one-off events in the noise floor — which is exactly why a
single-frame rule cannot be perfect, and why the roadmap suggests a temporal
defect map instead.

## 3. Fidelity against the original implementation

The original notebook produced `DATASET/target/` for 1133 frames. Applying this
implementation to the stored `bg_subtracted` frames and comparing with those
targets:

| Frame | Identical pixels | PSNR | SSIM | Energy ratio |
| --- | --- | --- | --- | --- |
| `cam1_image_1` | 99.983 % | 78.25 dB | 0.99999 | 1.002 |
| `cam3_image_40` | 99.995 % | 83.13 dB | 1.00000 | 1.007 |
| `cam6_image_1` | 99.987 % | 79.49 dB | 0.99999 | 1.002 |

The remaining differences (~200 pixels per frame, ≤10 ADU) come from the
original loop mutating its padded copy in place, and from a different
`skimage` version in the median filter.

## 4. Speed

1360 x 1040 frame, Python 3.14, Ryzen 7, single core:

| Implementation | Time per frame | Frames / s |
| --- | --- | --- |
| Per-pixel Python loop (reference) | 436 ms | 2.3 |
| Vectorised hot-pixel filter | 11.9 ms | 84 |
| Vectorised full chain | 39 ms | 25 |

Both timings are best-of-5 and are interleaved so that the two implementations see
the same machine load. Absolute numbers move by ±20 % with background load — the
first run of the day on an idle machine gave 348 ms / 8.9 ms / 32 ms (39×) — so
treat the ratio, not the milliseconds, as the result.

The vectorised result is bit-identical to the reference (0 differing pixels over
1 414 400). The original practice report quoted 1.2 frames/s for the whole
pipeline including disk I/O; the difference is mostly the removal of the Python
loop, not the removal of I/O.

## 5. What the 512 x 512 preprocessing did

The U-Net dataset was built by resizing 1360 x 1040 frames to 512 x 512 with
LANCZOS. That resize is not neutral:

| Property | Full resolution | After 512 x 512 resize |
| --- | --- | --- |
| Photon events (cam1 / cam3 / cam6) | 523 / 168 / 307 | 106 / 49 / 122 |
| Share of single-pixel events | 58.5 % | 12.9 % |
| Mean intensity | 0.152 ADU | 0.153 ADU |
| Aspect ratio | 1.31 | 1.00 (distorted 1.31x) |

A linear resize preserves the average brightness but changes the *shape* of the
signal: neighbouring photons merge into one blob, single-pixel events disappear
into their neighbours, and the whole image is stretched vertically by 31 %.

## 6. The U-Net baseline

Three predictions from `notebooks/legacy/02_unet_baseline.ipynb`:

| Sample | Input mean | Prediction mean | Target mean | Prediction SSIM |
| --- | --- | --- | --- | --- |
| 0 | 3.14 ADU | 9.12 ADU | 0.12 ADU | 0.092 |
| 1 | 3.21 ADU | 9.44 ADU | 0.17 ADU | 0.091 |
| 2 | 3.03 ADU | 9.29 ADU | 0.02 ADU | 0.082 |

The network learned a smooth grey haze at ~9 ADU: 230x the energy of the target
and 99.95 % of pixels non-zero, against 0.2–1.8 % in the target. It reproduced
the *location* of the beam and lost the photons. See
[`why-not-yolo.md`](why-not-yolo.md) for the analysis.
