# Photon frame cleaning

**English** · [Русский](README.ru.md)

[![tests](https://github.com/d1m40fficial/photon-denoising/actions/workflows/tests.yml/badge.svg)](https://github.com/d1m40fficial/photon-denoising/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/vectorised%20with-OpenCV-5c3ee8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Throughput](https://img.shields.io/badge/throughput-%E2%89%8825--30%20frames%2Fs-0969da)](#speed)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

Recovering sparse photon hits from 8-bit photon-counting camera frames: a static
sensor pedestal on every pixel, hot pixels that look exactly like photons, and a
real signal that covers about 1 % of the sensor.

A classical, fully vectorised pipeline does the job in **about 40 ms per
1360 × 1040 frame** on one CPU core. A U-Net baseline was trained on the same
data, failed for a measurable reason, and is kept in the repository as a
documented negative result.

![Before and after](assets/hero.png)

---

## The problem

Each frame contains three things that look similar in 8-bit grey:

| What | Signature | Measured |
| --- | --- | --- |
| Static pedestal | the same offset on every pixel, every frame | 3 ADU on 98.6 % of pixels |
| Hot pixels | bright dots, fixed coordinates, 1–6 pixels | ~18 000 pixels flagged per frame |
| Real photons | diffraction-limited spots, 2–4 pixels, 10–255 ADU | ~310 events per frame |

The raw frame is therefore 100 % non-zero and mostly grey — the photons are there,
but they are buried under a constant offset that is only 3 counts deep.

## Results at a glance

| | |
| --- | --- |
| Background estimate vs an independent exposure | **55.12 dB PSNR**, SSIM 0.9965 |
| Pixels left after cleaning | 100 % → 5.88 % → **1.16 %** |
| Photon events per frame | 26 034 → **310** |
| Energy kept above 50 ADU | **100.00 %** |
| Bit-exact agreement with the original implementation | **99.988 %** of pixels |
| Throughput | **≈25–30 frames/s** (≈35–40× faster than the reference loop) |

All numbers are reproducible from the shipped sample with
`scripts/evaluate.py` and `scripts/benchmark.py`; see [`docs/results.md`](docs/results.md).

## Why not deep learning?

This is the part of the project that took the longest and taught the most, so it
is documented in full: [`docs/why-not-yolo.md`](docs/why-not-yolo.md).

The short version:

* **Object detection is the wrong abstraction.** A photon has no boundary, a
  bounding box discards the intensity profile, and detection still leaves the
  reconstruction problem unsolved.
* **The U-Net was asked to learn a function that does not exist.** Its target was
  produced by the classical filter, which deletes *any* isolated bright pixel. For
  a patch containing one isolated bright pixel the correct answer is 0 whether it
  was a defect or a photon — so the network could only learn the conditional
  mean: a grey haze at 9.3 ADU against a target of 0.1 ADU.
* **The preprocessing had already destroyed the signal.** Building the dataset at
  512 × 512 removed 70 % of the photon events, cut the share of single-pixel
  events from 58 % to 13 %, and stretched the aspect ratio by 1.31×.

![U-Net failure](assets/unet_failure.png)

The pipeline that replaced it is classical, has no learned parameters, and its
error modes are measurable.

## How it works

![Pipeline](assets/pipeline.png)

1. **Background** — per-pixel median and standard deviation over a stack of
   frames. Pixels with `std > 50` or `median < 1` do not vote, so hot pixels and
   flickering pixels never become part of the background. Frames are processed in
   batches of 200, so a 1000-frame stack never has to fit in RAM.
2. **Subtract** — `clip(frame - background, 0, 255)`.
3. **Hot pixels** — delete a pixel if it is brighter than 1 ADU, the mean of its
   non-zero 3 × 3 neighbours is below 1 ADU, and it is more than 3× brighter than
   that mean.
4. **Clusters** — delete pixels that are 4× above the local mean *and* flat
   (within 15 % of a Gaussian-smoothed image). Photons are sharp, defect plateaus
   are not.
5. **Median filter** — `disk(1)`, to even out the remaining residue.

Full details in [`docs/method.md`](docs/method.md).

![Cleaning stages](assets/cleaning_stages.png)

## Validation

**Background.** The estimate is compared with a separate background exposure that
the algorithm never sees. The two agree to 55.12 dB PSNR / SSIM 0.9965 on the
shipped 24-frame stack, and to 55.79 dB / 0.9969 on the full 120-frame capture —
which reproduces the number from the original practice report.

![Background validation](assets/background_validation.png)

**Fidelity.** Run on the stored `bg_subtracted` frames, this implementation
reproduces the original notebook's `target` images on 99.988 % of pixels (78–83 dB
PSNR). The remaining ~200 pixels per frame come from an in-place mutation bug in
the original loop and a different `scikit-image` version.

**What the cleaning costs.** Total energy drops to 66.5 %, but the loss is
entirely in the 1–5 ADU noise floor: pixels above 50 ADU are untouched, and 99.94 %
of the energy above 10 ADU survives.

![Intensity profiles](assets/intensity_profile.png)

## Speed

| Implementation | Time per frame | Frames / s |
| --- | --- | --- |
| Per-pixel Python loop (reference) | 0.40–0.45 s | 2–3 |
| Vectorised hot-pixel filter | ~12 ms | ~85 |
| Vectorised full chain | 35–45 ms | 20–30 |

The vectorised output is **bit-identical** to the per-pixel reference: zero
differing pixels over 1 414 400.

![Benchmark](assets/benchmark.png)

## Quick start

```bash
git clone https://github.com/d1m40fficial/photon-denoising.git && cd photon-denoising
pip install -e ".[figures]"
```

Estimate a background and clean a frame:

```bash
photon-clean background --frames data/samples/frames --out background.png
photon-clean clean \
    --input data/samples/full/input/cam1_image_1.png \
    --background background.png \
    --out cleaned.png
```

Or from Python:

```python
from photon_clean import estimate_background_from_paths, subtract_background, clean_image
from photon_clean.io_utils import list_images, load_gray, save_gray

frames = list_images("data/samples/frames")
background, std_map = estimate_background_from_paths(frames)

frame = load_gray("data/samples/full/input/cam1_image_1.png")
cleaned = clean_image(subtract_background(frame, background))
save_gray(cleaned, "cleaned.png")
```

Build a dataset of `input / bg_subtracted / target` folders:

```bash
python scripts/build_dataset.py --frames data/samples/frames --out data/derived/sample_dataset
```

Regenerate every figure and number:

```bash
python scripts/evaluate.py            # results/metrics.json
python scripts/benchmark.py           # results/benchmark.json
python scripts/make_figures.py        # assets/*.png
python -m unittest discover -s tests  # 13 tests
```

## Repository layout

```
photon-denoising/
├── src/photon_clean/       library: background, cleaning, metrics, dataset, CLI
├── scripts/                evaluate, benchmark, build_dataset, make_figures
├── tests/                  unittest suite (13 tests, no network, < 1 s)
├── data/samples/           24 raw frames, 14 pairs, 3 full-res triples
├── data/legacy/unet/       the actual U-Net predictions
├── notebooks/legacy/       the two original notebooks (outputs cleared)
├── assets/                 every figure used in this README
├── docs/                   method, results, why-not-yolo
└── results/                metrics.json, benchmark.json
```

## Dataset

The full capture is 1133 frames from seven cameras (cam1–cam7) at 1360 × 1040.
The repository ships a 5 MB slice that reproduces everything above: see
[`data/README.md`](data/README.md).

## Limitations and roadmap

Honest list, because the interesting part of this project is where it stops
working:

* **Single-pixel photons cannot be separated from hot pixels.** Both are one bright
  pixel with nothing around them; the filter removes both. That is why the energy
  above 50 ADU is preserved but the 1–5 ADU bin loses 23 % — and why the network
  that tried to imitate the filter had nothing to learn.
* **A perfectly stable defect is absorbed into the background** by the median and
  never cleaned.
* **The method is single-frame and stateless** — it ignores the one property that
  actually separates the two classes.

Roadmap:

- [ ] **Temporal defect map.** Label pixels by persistence across a short stack (a
      defect recurs, a photon does not), build a mask once per session, and apply
      it to single frames. Only 2.5 % of currently deleted pixels recur in 6+ frames,
      so the current filter is mostly deleting noise — a temporal map would keep
      those photons.
- [ ] **Preserve full resolution** in any future ML experiment, and never resize
      with a changed aspect ratio.
- [ ] **Poisson-aware metrics.** MSE on a 1 %-sparse image is dominated by the
      background; event-level precision/recall is the metric that matters.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/method.md`](docs/method.md) | algorithms, parameters, complexity, failure modes |
| [`docs/results.md`](docs/results.md) | every measurement, with tables |
| [`docs/why-not-yolo.md`](docs/why-not-yolo.md) | YOLO and U-Net analysis |
| [`data/README.md`](data/README.md) | what is in the sample data and why |

## License

MIT — see [`LICENSE`](LICENSE).
