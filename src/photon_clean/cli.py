"""Command-line interface: ``python -m photon_clean <command>``."""

from __future__ import annotations

import argparse
import json
import os

from .background import estimate_background_from_paths, subtract_background
from .cleaning import CleanConfig, clean_image
from .dataset import build_dataset
from .io_utils import list_images, load_gray, save_gray


def _cmd_background(args: argparse.Namespace) -> int:
    paths = list_images(args.frames)
    if not paths:
        raise SystemExit(f"No images found in {args.frames}")

    background, std_map = estimate_background_from_paths(
        paths,
        batch_size=args.batch_size,
        brightness_threshold=args.brightness,
        std_threshold=args.std,
    )
    save_gray(background, args.out)
    if args.std_out:
        peak = float(std_map.max()) or 1.0
        save_gray((std_map * 255.0 / peak), args.std_out)

    share = float((background > args.brightness).mean())
    print(f"{len(paths)} frames -> {args.out}")
    print(f"background covers {share:.2%} of the sensor, mean {background.mean():.2f} ADU")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    image = load_gray(args.input)
    if args.background:
        image = subtract_background(image, load_gray(args.background))
    cleaned = clean_image(image, CleanConfig())
    save_gray(cleaned, args.out)
    kept = cleaned.sum() / max(int(image.sum()), 1)
    print(f"{args.input} -> {args.out} (energy kept {kept:.1%})")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    stats = build_dataset(
        args.frames,
        args.out,
        batch_size=args.batch_size,
        brightness_threshold=args.brightness,
        std_threshold=args.std,
        limit=args.limit,
    )
    if args.stats:
        payload = {
            "frames": stats.frames,
            "background_pixels": stats.background_pixels,
            "background_share": stats.background_share,
            "mean_energy_in": stats.mean_energy_in,
            "mean_energy_bg": stats.mean_energy_bg,
            "mean_energy_out": stats.mean_energy_out,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.stats)), exist_ok=True)
        with open(args.stats, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photon_clean",
        description="Clean photon-counting camera frames: static background and hot pixels.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bg = sub.add_parser("background", help="estimate the static background from a frame stack")
    bg.add_argument("--frames", required=True, help="directory with the frame stack")
    bg.add_argument("--out", required=True, help="output PNG for the background estimate")
    bg.add_argument("--std-out", default=None, help="optional PNG for the std map")
    bg.add_argument("--batch-size", type=int, default=200)
    bg.add_argument("--brightness", type=int, default=1)
    bg.add_argument("--std", type=float, default=50.0)
    bg.set_defaults(func=_cmd_background)

    clean = sub.add_parser("clean", help="clean one frame")
    clean.add_argument("--input", required=True)
    clean.add_argument("--background", default=None, help="background PNG to subtract first")
    clean.add_argument("--out", required=True)
    clean.set_defaults(func=_cmd_clean)

    build = sub.add_parser("build", help="build an input/bg_subtracted/target dataset")
    build.add_argument("--frames", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--stats", default=None, help="optional JSON with summary statistics")
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--batch-size", type=int, default=200)
    build.add_argument("--brightness", type=int, default=1)
    build.add_argument("--std", type=float, default=50.0)
    build.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
