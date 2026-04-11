# NLP Project — Setup Guide Spring 2026

## Step-by-Step AWS Setup & Fine-Tuning Guide
### Arabic Handwritten Text Recognition with Qwen2.5-VL + LoRA
**MAI 656 — Natural Language Processing**
**Canadian University Dubai**
**Spring 2026**

---

## Quick Reference

| Item | Value |
|------|-------|
| AWS Instance | g5.xlarge (NVIDIA A10G, 24GB VRAM) |
| AMI | Deep Learning OSS PyTorch Ubuntu 22 |
| Region | us-east-1 (cheapest for GPU) |
| Storage | 100 GB gp3 |
| Cost | ~$1.00/hr on-demand |
| Training time | ~2–4 hours |
| Total cost per run | ~$3–5 |

> **WARNING**
> Always terminate your instance when done. A running g5.xlarge costs ~$24/day.

---

## Contents

1. [AWS Account Setup (One-Time)](#1-aws-account-setup-one-time)
   - 1.1 [Create an AWS Account](#11-create-an-aws-account)
   - 1.2 [Request GPU Quota](#12-request-gpu-quota)
   - 1.3 [Create a Key Pair](#13-create-a-key-pair)
   - 1.4 [Create a Security Group](#14-create-a-security-group)
2. [Launch a GPU Instance](#2-launch-a-gpu-instance)
   - 2.1 [Launch Instance](#21-launch-instance)
   - 2.2 [Connect via SSH](#22-connect-via-ssh)
   - 2.3 [Verify GPU](#23-verify-gpu)
3. [Environment Setup](#3-environment-setup)
   - 3.1 [Create a Virtual Environment](#31-create-a-virtual-environment)
   - 3.2 [Install Dependencies](#32-install-dependencies)
   - 3.3 [Set Up Your Project](#33-set-up-your-project)
   - 3.4 [Upload Your Code](#34-upload-your-code)
4. [Stage 1 — Label Data with GPT (Teacher)](#4-stage-1--label-data-with-gpt-teacher)
   - 4.1 [Set Your OpenAI API Key](#41-set-your-openai-api-key)
   - 4.2 [Create the GPT Labeling Script](#42-create-the-gpt-labeling-script)
   - 4.3 [Run the Labeling](#43-run-the-labeling)
5. [Stage 2 — Transform Data (GPT to Qwen Format)](#5-stage-2--transform-data-gpt-format--qwen-format)
   - 5.1 [Understand the Format Difference](#51-understand-the-format-difference)
   - 5.2 [Create the Transformation Script](#52-create-the-transformation-script)
   - 5.3 [Run the Transformation](#53-run-the-transformation)
6. [Stage 3 — Fine-Tuning with LoRA](#6-stage-3--fine-tuning-with-lora)
   - 6.1 [Create the Configuration](#61-create-the-configuration)
   - 6.2 [Create the Training Script](#62-create-the-training-script)
   - 6.3 [Run Training](#63-run-training)
7. [Stage 4 — Evaluation](#7-stage-4--evaluation)
   - 7.1 [Create the Evaluation Script](#71-create-the-evaluation-script)
   - 7.2 [Run Evaluation](#72-run-evaluation)
8. [Stage 5 — Inference Server (Optional Demo)](#8-stage-5--inference-server-optional-demo)
   - 8.1 [Launch vLLM Server](#81-launch-vllm-server)
   - 8.2 [Test the Server](#82-test-the-server)
9. [TERMINATE Your Instance!](#9-terminate-your-instance)
   - 9.1 [Option A: From the AWS Console](#91-option-a-from-the-aws-console)
   - 9.2 [Option B: From the Command Line](#92-option-b-from-the-command-line)
   - 9.3 [Option C: Stop (Not Terminate) to Save Your Work](#93-option-c-stop-not-terminate-to-save-your-work)
10. [Troubleshooting Cheat Sheet](#10-troubleshooting-cheat-sheet)
11. [Downloading Results](#11-downloading-results)
12. [Quick Command Reference](#12-quick-command-reference)

---

## 1 AWS Account Setup (One-Time)

### 1.1 Create an AWS Account

1. Go to https://aws.amazon.com and create an account
2. You will need a credit card (you will only be charged for what you use)
3. New accounts get some free-tier credits, but GPU instances are NOT free-tier

### 1.2 Request GPU Quota

By default, new AWS accounts have zero GPU quota. You must request it:

1. Go to Service Quotas → Amazon EC2
2. Search for: **Running On-Demand G and VT instances**
3. Click **Request quota increase**
4. Request a value of **4** (g5.xlarge needs 4 vCPUs)
5. Wait for approval (usually 1–24 hours)

> **IMPORTANT**
> Without this step, you will get an error when trying to launch your instance.

### 1.3 Create a Key Pair

1. Go to EC2 → Key Pairs → Create key pair
2. Name it: `nlp-project`
3. Key type: RSA
4. Format: `.pem`
5. Download and save the file. Do not lose it.
6. Set permissions:

```bash
chmod 400 ~/Downloads/nlp-project.pem
```

### 1.4 Create a Security Group

1. Go to EC2 → Security Groups → Create security group
2. Name: `nlp-gpu-server`
3. Add inbound rules:
   - SSH (port 22) — your IP only
   - Custom TCP (port 8000) — your IP only (for inference server)
4. Save

---

## 2 Launch a GPU Instance

### 2.1 Launch Instance

1. Go to EC2 → Launch Instance
2. Configure:

| Setting | Value |
|---------|-------|
| Name | nlp-finetuning |
| AMI | Search for "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" — choose Ubuntu 22.04 |
| Instance type | g5.xlarge |
| Key pair | nlp-project |
| Security group | nlp-gpu-server |
| Storage | Change to 100 GB, type gp3 |

3. Click **Launch instance**
4. Note the Instance ID — you will need it to terminate later

### 2.2 Connect via SSH

Wait ~2 minutes for the instance to boot, then:

```bash
# Get the public IP from the EC2 console
ssh -i ~/Downloads/nlp-project.pem ubuntu@<PUBLIC_IP>
```

### 2.3 Verify GPU

```bash
nvidia-smi
```

You should see: `NVIDIA A10G | 24GB VRAM`.

If you see an error, the AMI may need a reboot:

```bash
sudo reboot
# Wait 1 minute, then SSH again
```

---

## 3 Environment Setup

### 3.1 Create a Virtual Environment

**Option A — Conda (recommended on the AWS Deep Learning AMI)**

The Deep Learning AMI ships with conda pre-installed. Use this on the GPU instance.

```bash
# Create the environment
conda create -n nlp python=3.11 -y

# Activate
conda activate nlp

# Deactivate when done
conda deactivate
```

---

**Option B — Python `venv` (for local machines or plain Ubuntu)**

Use this if you are working locally (Stage 1 labeling, data prep) or on a machine without conda.

**Linux / macOS**

```bash
# Create the virtual environment in the project directory
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Deactivate when done
deactivate
```

**Windows (Command Prompt)**

```cmd
# Create the virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate.bat

# Deactivate when done
deactivate
```

**Windows (PowerShell)**

```powershell
# Create the virtual environment
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Deactivate when done
deactivate
```

> **NOTE (Windows PowerShell)**
> If you get a script execution error, run this once to allow local scripts:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3.2 Install Dependencies

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers>=4.40 peft>=0.10 bitsandbytes>=0.43 accelerate>=0.30 datasets Pillow PyMuPDF openai trl wandb

# For inference server (install later, or now)
pip3 install vllm>=0.4
```

### 3.3 Set Up Your Project

```bash
mkdir -p ~/project
cd ~/project

# Create the directory structure
mkdir -p data/raw data/processed data/train data/eval
mkdir -p models/lora_adapter
mkdir -p logs
```

### 3.4 Upload Your Code

From your local machine, upload your scripts:

```bash
scp -i ~/Downloads/nlp-project.pem -r ./your_code/* ubuntu@<PUBLIC_IP>:~/project/
```

Or clone from your GitHub repo:

```bash
cd ~/project
git clone https://github.com/YOUR_REPO.git .
```

---

## 4 Stage 1 — Label Data with GPT (Teacher)

> **TIP**
> This stage can be done on your local machine or Colab to save GPU costs. You only need the GPU instance for training (Stage 3).

### 4.1 Set Your OpenAI API Key

**Linux / macOS**

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Windows (Command Prompt)**

```cmd
set OPENAI_API_KEY=sk-your-key-here
```

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

> **NOTE**
> These commands set the key only for the current shell session. Use section 4.1.1 below for a persistent approach.

### 4.1.1 Alternative: Store the API Key in a `.env` File

Instead of exporting the key every session, store it in a `.env` file in your project directory. This approach works on all platforms and persists across sessions.

**Step 1 — Create the `.env` file**

```bash
# Linux / macOS / Git Bash on Windows
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

Or create the file manually with any text editor and add:

```
OPENAI_API_KEY=sk-your-key-here
```

**Step 2 — Add `.env` to `.gitignore` (IMPORTANT)**

```bash
echo '.env' >> .gitignore
```

Never commit your API key to git. This step is mandatory.

**Step 3 — Load the key at runtime**

Install `python-dotenv`:

```bash
pip3 install python-dotenv
```

Then add these two lines to the top of any script that needs the key:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env and sets environment variables
```

After this, `os.environ["OPENAI_API_KEY"]` will work as normal.

**Alternative — Load from the shell (no code change needed)**

```bash
# Linux / macOS
set -a && source .env && set +a

# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
```

### 4.2 Create the GPT Labeling Script

Create `collect_data.py`:

```python
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
```

### 4.3 Run the Labeling

```bash
python3 collect_data.py
```

Budget estimate: ~200 images at ~$0.01–0.03 each = $2–6 total.

---

## 5 Stage 2 — Transform Data (GPT Format → Qwen Format)

### 5.1 Understand the Format Difference

GPT receives images directly. Qwen2.5-VL also receives images, but expects a specific chat format:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an Arabic handwriting recognition system..."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,..."}}
            ]
        },
        {
            "role": "assistant",
            "content": "{\"transcription\": \"Arabic text\", \"confidence\": \"high\"}"
        }
    ]
}
```

### 5.2 Create the Transformation Script

Create `transform_data.py`:

```python
import json
import base64
import random
from pathlib import Path
from PIL import Image
import io

# IMPORTANT: These must match between training and inference!
MIN_PIXELS = 4 * 28 * 28    # 3,136
MAX_PIXELS = 128 * 28 * 28  # 100,352
DPI = 100

def simplify_gpt_output(raw_response: str) -> str:
    """Strip GPT's output to essential fields only."""
    try:
        data = json.loads(raw_response)
        simplified = {
            "transcription": data.get("transcription", ""),
            "confidence": data.get("confidence", "unknown")
        }
        return json.dumps(simplified, ensure_ascii=False)
    except json.JSONDecodeError:
        return None

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string, respecting pixel bounds."""
    img = Image.open(image_path)
    w, h = img.size
    total_pixels = w * h

    if total_pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / total_pixels) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif total_pixels < MIN_PIXELS:
        scale = (MIN_PIXELS / total_pixels) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def transform_sample(sample: dict) -> dict:
    """Convert a GPT-labeled sample to Qwen training format."""
    image_path = sample["image_path"]
    gpt_response = sample["gpt_response"]

    simplified = simplify_gpt_output(gpt_response)
    if simplified is None:
        return None

    try:
        img_b64 = image_to_base64(image_path)
    except Exception:
        return None

    return {
        "messages": [
            {"role": "system",
             "content": "You are an Arabic handwriting recognition system. "
                        "Read the handwritten Arabic text in this image and "
                        "return the transcription as JSON."},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]},
            {"role": "assistant", "content": simplified}
        ]
    }

def main():
    input_file = "data/gpt_labels.jsonl"
    train_file = "data/train/train.jsonl"
    eval_file = "data/eval/eval.jsonl"

    samples = []
    with open(input_file) as f:
        for line in f:
            sample = json.loads(line)
            transformed = transform_sample(sample)
            if transformed:
                samples.append(transformed)

    print(f"Successfully transformed {len(samples)} samples")

    # Split 80/20
    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    for filepath, data in [(train_file, train_samples),
                            (eval_file, eval_samples)]:
        with open(filepath, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_samples)} samples")
    print(f"Eval: {len(eval_samples)} samples")

if __name__ == "__main__":
    main()
```

### 5.3 Run the Transformation

```bash
python3 transform_data.py
```

---

## 6 Stage 3 — Fine-Tuning with LoRA

> **WARNING**
> This is the most critical stage. Run this on the GPU instance (g5.xlarge).

### 6.1 Create the Configuration

Create `config.py`:

```python
# ============================================================
# Configuration -- All hyperparameters in one place
# ============================================================

# Model
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SEQ_LENGTH = 8192  # CRITICAL: Do NOT set below 8192!

# LoRA
LORA_R = 32           # Rank (max for 24 GB GPU with this model)
LORA_ALPHA = 64       # Scaling factor (2x rank)
LORA_DROPOUT = 0.05   # Regularization
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # Feed-forward
]

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2  # Effective batch = 2
LEARNING_RATE = 2e-5
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1

# Image processing (MUST match inference settings!)
MIN_PIXELS = 4 * 28 * 28    # 3,136
MAX_PIXELS = 128 * 28 * 28  # 100,352
DPI = 100

# Quantization
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = "bfloat16"
DOUBLE_QUANT = True

# Paths
TRAIN_DATA = "data/train/train.jsonl"
EVAL_DATA = "data/eval/eval.jsonl"
OUTPUT_DIR = "models/lora_adapter"
LOG_DIR = "logs"
```

### 6.2 Create the Training Script

Create `train.py`:

```python
import torch
import json
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import config

# 1. Load Model with 4-bit Quantization
print("Loading model with 4-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=config.LOAD_IN_4BIT,
    bnb_4bit_quant_type=config.QUANT_TYPE,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=config.DOUBLE_QUANT,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config.BASE_MODEL,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

processor = AutoProcessor.from_pretrained(
    config.BASE_MODEL,
    min_pixels=config.MIN_PIXELS,
    max_pixels=config.MAX_PIXELS,
)

# 2. Apply LoRA
print("Applying LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load Dataset
class ArabicHTRDataset(Dataset):
    def __init__(self, data_path, processor, max_length):
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=text, return_tensors="pt",
            max_length=self.max_length,
            truncation=True, padding="max_length",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}

print("Loading datasets...")
train_dataset = ArabicHTRDataset(
    config.TRAIN_DATA, processor, config.MAX_SEQ_LENGTH)
eval_dataset = ArabicHTRDataset(
    config.EVAL_DATA, processor, config.MAX_SEQ_LENGTH)

print(f"Training: {len(train_dataset)}, Eval: {len(eval_dataset)}")

if len(train_dataset) == 0:
    raise ValueError(
        "NO TRAINING SAMPLES! MAX_SEQ_LENGTH is probably too low.")

# 4. Training
print("Starting training...")
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
    learning_rate=config.LEARNING_RATE,
    lr_scheduler_type=config.LR_SCHEDULER,
    warmup_ratio=config.WARMUP_RATIO,
    bf16=True,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    logging_dir=config.LOG_DIR,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
)
trainer.train()

# 5. Save the LoRA Adapter
print(f"Saving LoRA adapter to {config.OUTPUT_DIR}...")
model.save_pretrained(config.OUTPUT_DIR)
processor.save_pretrained(config.OUTPUT_DIR)
print("Training complete!")
```

### 6.3 Run Training

```bash
cd ~/project
conda activate nlp
nvidia-smi  # Check GPU before starting
python3 train.py 2>&1 | tee logs/training.log
```

Expected output:

```
Loading model with 4-bit quantization...
Applying LoRA...
trainable params: 159,907,840 || all params: 8,289,003,520 || trainable%: 1.93%
Training samples: ~160
Evaluation samples: ~40
Starting training...
{'loss': 2.1234, 'learning_rate': 2e-05, 'epoch': 0.12}
...
Training complete!
Adapter size: ~380 MB
```

Training should take ~2–4 hours. Monitor with:

```bash
# In another SSH session:
tail -f ~/project/logs/training.log
nvidia-smi -l 10  # GPU usage every 10 seconds
```

---

## 7 Stage 4 — Evaluation

### 7.1 Create the Evaluation Script

Create `evaluate.py`:

```python
import torch
import json
import time
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor, BitsAndBytesConfig,
)
from peft import PeftModel
import config

# Load Model + LoRA Adapter
print("Loading base model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config.BASE_MODEL,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16, device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
model.eval()

processor = AutoProcessor.from_pretrained(
    config.BASE_MODEL,
    min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS,
)

# Robust JSON Parsing
def parse_output(raw_text: str) -> dict:
    """Try multiple strategies to extract valid JSON."""
    text = raw_text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code blocks
    if "'''" in text:
        text = text.split("'''json")[-1].split("'''")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Truncation search
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            try:
                return json.loads(text[:i+1])
            except json.JSONDecodeError:
                continue

    # Strategy 4: Auto-complete
    for suffix in ['"}', '"]}', '"}]', '"}]}']:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    return None

# Metrics
def char_error_rate(prediction: str, reference: str) -> float:
    import difflib
    if not reference:
        return 0.0 if not prediction else 1.0
    matcher = difflib.SequenceMatcher(None, reference, prediction)
    distance = sum(max(j2 - j1, i2 - i1) for tag, i1, i2, j1, j2
                   in matcher.get_opcodes() if tag != 'equal')
    return distance / max(len(reference), 1)

def word_error_rate(prediction: str, reference: str) -> float:
    ref_words = reference.split()
    pred_words = prediction.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    import difflib
    matcher = difflib.SequenceMatcher(None, ref_words, pred_words)
    distance = sum(max(j2 - j1, i2 - i1) for tag, i1, i2, j1, j2
                   in matcher.get_opcodes() if tag != 'equal')
    return distance / max(len(ref_words), 1)

# Run Evaluation
print("Loading eval dataset...")
eval_samples = []
with open(config.EVAL_DATA) as f:
    for line in f:
        eval_samples.append(json.loads(line))

# Token suppression
suppress = ["<tool_call>", "<|tool_call|>",
            "<tool_response>", "<|tool_response|>"]
bad_words_ids = []
for token in suppress:
    ids = processor.tokenizer.encode(token, add_special_tokens=False)
    if ids:
        bad_words_ids.append(ids)

results = []
for i, sample in enumerate(eval_samples):
    messages = sample["messages"]
    expected = messages[-1]["content"]
    input_messages = messages[:-1]

    text = processor.apply_chat_template(
        input_messages, tokenize=False, add_generation_prompt=True)

    # JSON prefix forcing
    json_prefix = '{"transcription":"'
    text += json_prefix

    inputs = processor(text=text, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            temperature=0.1, do_sample=False,
            bad_words_ids=bad_words_ids if bad_words_ids else None)
    inference_time = time.time() - start_time

    generated = processor.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)
    generated = json_prefix + generated

    pred_parsed = parse_output(generated)
    ref_parsed = parse_output(expected)

    pred_text = pred_parsed.get("transcription", "") if pred_parsed else ""
    ref_text = ref_parsed.get("transcription", "") if ref_parsed else ""

    cer = char_error_rate(pred_text, ref_text)
    wer = word_error_rate(pred_text, ref_text)
    exact = pred_text.strip() == ref_text.strip()

    results.append({"cer": cer, "wer": wer,
                    "exact_match": exact,
                    "valid_json": pred_parsed is not None,
                    "inference_time": inference_time})

    print(f"[{i+1}/{len(eval_samples)}] CER={cer:.3f} "
          f"WER={wer:.3f} Time={inference_time:.1f}s")

# Summary
avg_cer = sum(r["cer"] for r in results) / len(results)
avg_wer = sum(r["wer"] for r in results) / len(results)
exact_rate = sum(r["exact_match"] for r in results) / len(results) * 100
json_rate = sum(r["valid_json"] for r in results) / len(results) * 100
avg_time = sum(r["inference_time"] for r in results) / len(results)

print(f"\nAvg CER: {avg_cer:.4f}")
print(f"Avg WER: {avg_wer:.4f}")
print(f"Exact match: {exact_rate:.1f}%")
print(f"Valid JSON: {json_rate:.1f}%")
print(f"Avg time: {avg_time:.1f}s")

with open("logs/eval_results.json", "w") as f:
    json.dump({"avg_cer": avg_cer, "avg_wer": avg_wer,
               "exact_match_rate": exact_rate,
               "valid_json_rate": json_rate,
               "avg_inference_time": avg_time}, f, indent=2)
```

### 7.2 Run Evaluation

```bash
python3 evaluate.py 2>&1 | tee logs/eval.log
```

---

## 8 Stage 5 — Inference Server (Optional Demo)

### 8.1 Launch vLLM Server

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules "arabic-ocr=models/lora_adapter" \
    --served-model-name "arabic-ocr-v1" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 --dtype auto
```

> **IMPORTANT**
> `--max-lora-rank 32` MUST match your adapter's rank. vLLM defaults to 16, which will crash with: `ValueError: LoRA rank 32 is greater than max_lora_rank 16`

### 8.2 Test the Server

From another terminal:

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test inference
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "arabic-ocr-v1",
        "messages": [
            {"role": "system",
             "content": "Read the Arabic handwriting and return JSON."},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,YOUR_BASE64"}}
            ]}
        ],
        "max_tokens": 512
    }'
```

---

## 9 TERMINATE Your Instance!

> **WARNING**
> This is the most important step. A forgotten g5.xlarge costs ~$24/day (~$720/month).

### 9.1 Option A: From the AWS Console

1. Go to EC2 → Instances
2. Select your instance
3. Actions → Instance State → Terminate

### 9.2 Option B: From the Command Line

```bash
aws ec2 terminate-instances --instance-ids i-YOUR_INSTANCE_ID --region us-east-1
```

### 9.3 Option C: Stop (Not Terminate) to Save Your Work

If you want to continue later without re-uploading data:

```bash
aws ec2 stop-instances --instance-ids i-YOUR_INSTANCE_ID --region us-east-1
# You still pay for storage (~$0.08/GB/month = ~$8/month for 100 GB)

# Restart later:
aws ec2 start-instances --instance-ids i-YOUR_INSTANCE_ID --region us-east-1
```

---

## 10 Troubleshooting Cheat Sheet

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA out of memory | LoRA rank too high or images too large | Reduce LORA_R to 16, reduce MAX_PIXELS |
| Training completed with 0 steps | MAX_SEQ_LENGTH too low | Set to 8192 or higher |
| Model outputs only `<\|im_end\|>` | Model learned to stop early | Use JSON prefix forcing in evaluation |
| All JSON outputs invalid | Model echoing input text | Simplify output format, use robust parser |
| `python: command not found` | AMI has python3 not python | Always use python3 |
| LoRA rank 32 > max_lora_rank 16 | vLLM default max rank is 16 | Add `--max-lora-rank 32` |
| Blurry text in images | DPI too low (default 72) | Use DPI=100 in config |
| Results differ train vs. inference | Different image settings | Same DPI, min/max pixels in both |
| InsufficientInstanceCapacity | No GPU available in region | Try a different AZ or region |
| vCPU quota exceeded | Old instance still shutting down | Wait 5 min, then try again |
| AccessDeniedException | Missing IAM permissions | Check your IAM policy has EC2 access |

---

## 11 Downloading Results

Before terminating, download your results to your local machine:

```bash
scp -i ~/Downloads/nlp-project.pem -r \
    ubuntu@<PUBLIC_IP>:~/project/models/lora_adapter ./results/

scp -i ~/Downloads/nlp-project.pem -r \
    ubuntu@<PUBLIC_IP>:~/project/logs/ ./results/
```

---

## 12 Quick Command Reference

```bash
# SSH into instance
ssh -i ~/Downloads/nlp-project.pem ubuntu@<PUBLIC_IP>

# Check GPU
nvidia-smi

# Activate environment
conda activate nlp

# Run the full pipeline
cd ~/project
python3 collect_data.py    # Stage 1: Label with GPT
python3 transform_data.py  # Stage 2: Transform data
python3 train.py           # Stage 3: Fine-tune (2-4 hours)
python3 evaluate.py        # Stage 4: Evaluate

# Monitor training
tail -f logs/training.log
nvidia-smi -l 10

# TERMINATE when done!
aws ec2 terminate-instances --instance-ids <ID> --region us-east-1
```
