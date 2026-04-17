import os

INPUT_PATH  = "sample_training_data/gpt_labels.jsonl"
OUTPUT_PATH = "sample_training_data"
START_LINE  = 1    # 1-based, inclusive
END_LINE    = 100  # 1-based, inclusive

def split_jsonl(input_path, output_path, start, end):
    base = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base, input_path)
    output_path = os.path.join(base, output_path)

    # if output_path is a directory, auto-generate filename
    if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{stem}_{start}_{end}.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    written = 0
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            if i < start:
                continue
            if i > end:
                break
            outfile.write(line)
            written += 1

    print(f"Wrote {written} lines (lines {start}-{end}) to {output_path}")

if __name__ == "__main__":
    split_jsonl(INPUT_PATH, OUTPUT_PATH, START_LINE, END_LINE)
