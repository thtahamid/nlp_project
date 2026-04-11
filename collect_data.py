from dotenv import load_dotenv
load_dotenv()  # reads .env and sets environment variables

import openai
import base64
import json
import os
from pathlib import Path

client = openai.OpenAI()

SYSTEM_PROMPT = """You are an Arabic handwriting recognition system.
Read the handwritten Arabic text in this image and transcribe it exactly
as written. Return your response as JSON:
{
    "transcription": "the Arabic text you read",
    "confidence": "high/medium/low"
}
Return ONLY the JSON, no other text."""

def label_image(image_path: str) -> dict:
    """Send an image to GPT and get the transcription."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{image_data}"
                }}
            ]}
        ],
        max_tokens=2048,
        temperature=0.0
    )

    return {
        "image_path": image_path,
        "gpt_response": response.choices[0].message.content,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
    }

def main():
    image_dir = "data/raw"
    output_file = "data/gpt_labels.jsonl"

    images = sorted(Path(image_dir).glob("*.png")) + \
             sorted(Path(image_dir).glob("*.jpg"))

    print(f"Found {len(images)} images to label")

    with open(output_file, "a") as f:
        for i, img_path in enumerate(images):
            print(f"[{i+1}/{len(images)}] Labeling {img_path.name}...")
            try:
                result = label_image(str(img_path))
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    print(f"Done! Labels saved to {output_file}")

if __name__ == "__main__":
    main()