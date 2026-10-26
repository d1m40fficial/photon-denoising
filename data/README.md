# Sample data

The original capture is 1133 frames from seven photon-counting cameras. Shipping
all of it would mean ~8 GB of BMP files, so this repository keeps a small,
self-contained slice that is enough to reproduce every figure and number in the
README.

```
data/
├── samples/
│   ├── frames/                 24 consecutive raw frames (cam1), 1360 x 1040 PNG
│   ├── reference_background.png  an independent background exposure (fon1_0.bmp)
│   ├── pairs/input/            14 downscaled 512 x 512 frames, 2 per camera
│   ├── pairs/target/           their cleaned counterparts
│   └── full/
│       ├── input/              3 raw frames at full resolution
│       ├── bg_subtracted/      the same 3 frames after background subtraction
│       └── target/             the same 3 frames after the full cleaning chain
└── legacy/
    └── unet/                   6 PNGs: input / predicted / target for 3 samples
```

## What each part is for

| Folder | Used by | Why it is here |
| --- | --- | --- |
| `samples/frames` | background estimation, cleaning, benchmark | a 24-frame stack is enough for a stable per-pixel median |
| `samples/reference_background.png` | `scripts/evaluate.py` | independent ground truth for the background estimate |
| `samples/pairs` | dataset examples | shows the 512 x 512 pairs the U-Net was trained on |
| `samples/full` | fidelity and downscaling analysis | full-resolution input / intermediate / output of the original pipeline |
| `legacy/unet` | U-Net failure figure | the actual predictions produced by the failed baseline |

## Format

* 8-bit grayscale PNG, no alpha channel.
* Full frames are 1360 x 1040 (aspect ratio 1.31).
* The 512 x 512 files in `pairs/` are **not** a faithful resize: the original
  preprocessing squashed 1360 x 1040 to a square, distorting the aspect ratio by
  1.31x. This is documented on purpose, because it is one of the reasons the
  U-Net baseline could not work (`docs/why-not-yolo.md`).
* Values are raw ADU. A sensor bias of about 3 ADU sits on every pixel, so the
  mean of a raw frame is ~3.27 ADU even where nothing was detected.

## Cameras

The full dataset contains `cam1` … `cam7` with 120 / 140 / 180 / 73 / 230 / 270 /
120 frames respectively. `cam1` is the stack shipped in `samples/frames`
(24 of its 120 frames).

## Provenance

The frames were captured at the Institute of Metal Physics (Ural Branch of the
Russian Academy of Sciences) and processed during a university practice project
in 2025. Only the frames needed for the demonstration are redistributed here;
the raw BMP/PNG archives and the administrative documents are not part of this
repository.
