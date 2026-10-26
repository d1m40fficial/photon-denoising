"""Build ``input / bg_subtracted / target`` folders from a stack of raw frames.

Example
-------
    python scripts/build_dataset.py \
        --frames data/samples/frames \
        --out data/derived/sample_dataset \
        --stats results/sample_dataset.json

The three folders are exactly the ones the legacy U-Net experiment was trained
on: the network learned ``input -> target``, i.e. "raw frame -> heuristic output".
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import ROOT  # noqa: F401  (adds src/ to sys.path)

from photon_clean import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", required=True, help="directory with raw frames")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--stats", default=None, help="optional JSON with summary statistics")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N frames")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--brightness", type=int, default=1)
    parser.add_argument("--std", type=float, default=50.0)
    args = parser.parse_args(argv)

    stats = build_dataset(
        args.frames,
        args.out,
        batch_size=args.batch_size,
        brightness_threshold=args.brightness,
        std_threshold=args.std,
        limit=args.limit,
    )

    if args.stats:
        import json
        from pathlib import Path

        path = Path(args.stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "frames": stats.frames,
                    "background_pixels": stats.background_pixels,
                    "background_share": stats.background_share,
                    "mean_energy_in": stats.mean_energy_in,
                    "mean_energy_bg": stats.mean_energy_bg,
                    "mean_energy_out": stats.mean_energy_out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"statistics written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
