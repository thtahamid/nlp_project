#!/usr/bin/env python3
"""Convert all TIF/TIFF files in data/raw to PNG and save them in data/raw_png."""

from pathlib import Path
from PIL import Image

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/raw_png")


def convert_tif_to_png() -> None:
    if not INPUT_DIR.is_dir():
        raise ValueError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tif_files = list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.tiff"))
    if not tif_files:
        print("No TIF/TIFF files found.")
        return

    print(f"Found {len(tif_files)} file(s). Converting...")
    for tif_file in tif_files:
        out_file = OUTPUT_DIR / (tif_file.stem + ".png")
        with Image.open(tif_file) as img:
            img.save(out_file, format="PNG")
        print(f"  {tif_file.name} -> {out_file.name}")

    print("Done.")


if __name__ == "__main__":
    convert_tif_to_png()
