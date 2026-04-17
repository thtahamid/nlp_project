import os
import shutil

INPUT_PATH  = "data/raw_png"
OUTPUT_PATH = "sample_training_data/"
START_IDX   = 1    # 1-based, inclusive
END_IDX     = 100  # 1-based, inclusive

def split_png(input_path, output_path, start, end):
    base = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(base, input_path)
    output_path = os.path.join(base, output_path)

    png_files = sorted(
        f for f in os.listdir(input_path)
        if f.lower().endswith(".png")
    )

    selected = png_files[start - 1 : end]

    stem = os.path.basename(input_path.rstrip("/\\"))
    out_dir = os.path.join(output_path, f"{stem}_{start}_{end}")
    os.makedirs(out_dir, exist_ok=True)

    for fname in selected:
        shutil.copy2(os.path.join(input_path, fname), os.path.join(out_dir, fname))

    print(f"Copied {len(selected)} PNG(s) (indices {start}-{end}) to {out_dir}")

if __name__ == "__main__":
    split_png(INPUT_PATH, OUTPUT_PATH, START_IDX, END_IDX)
