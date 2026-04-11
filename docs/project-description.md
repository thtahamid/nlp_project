# NLP Final Project

## Knowledge Distillation from GPT to Qwen for Arabic Handwritten Text Recognition

| Field | Details |
|---|---|
| Course | MAI 656 — Natural Language Processing |
| Institution | Canadian University Dubai |
| Semester | Spring 2026 |
| Due Date | TBD |
| Team Size | 2–3 students |
| Weight | 35% (Report 20% + Presentation 10% + Code 5%) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Matters](#2-why-this-matters)
3. [Dataset](#3-dataset)
4. [Project Stages](#4-project-stages)
   - [Stage 1: Data Collection & GPT Labeling](#41-stage-1-data-collection--gpt-labeling-the-teacher)
   - [Stage 2: Data Transformation](#42-stage-2-data-transformation-gpt-format--qwen-format)
   - [Stage 3: Fine-Tuning with LoRA](#43-stage-3-fine-tuning-with-lora-the-student)
   - [Stage 4: Evaluation](#44-stage-4-evaluation)
   - [Stage 5: Inference Server Deployment](#45-stage-5-inference-server-deployment)
   - [Stage 6: Downstream NLP Analysis](#46-stage-6-downstream-nlp-analysis)
   - [Stage 7: Baseline Comparison](#47-stage-7-baseline-comparison)
   - [Stage 8: Ethical and Trend Analysis](#48-stage-8-ethical-and-trend-analysis)
5. [Compute Resources](#5-compute-resources)
6. [Deliverables](#6-deliverables)
7. [Common Pitfalls & Tips](#7-common-pitfalls--tips)
8. [Grading Rubric](#8-grading-rubric)
9. [Bonus Challenges](#9-bonus-challenges-extra-credit)
10. [Required Libraries](#10-required-libraries)
11. [Getting Started Checklist](#11-getting-started-checklist)

---

## 1 Project Overview

Large commercial models like OpenAI's GPT-5.2 can read handwritten Arabic text from images with high accuracy — but they are expensive, slow, and require sending data to external servers. In this project, you will **distill** that capability into a smaller, open-source model (Qwen2.5-VL-7B) that can run locally on a single GPU.

### 1.1 Your Goal

Build an end-to-end pipeline that:

1. Uses **GPT-5.2** to generate high-quality transcriptions of Arabic handwritten text images (creating your training data)
2. **Fine-tunes Qwen2.5-VL-7B-Instruct** using LoRA to replicate GPT's transcription ability
3. Applies **downstream NLP tasks** (NER, POS tagging) to the transcribed Arabic text
4. **Compares the distilled model** against classical baselines (CNN+BiLSTM+CTC or TrOCR)
5. **Evaluates** how closely your fine-tuned model matches GPT's output
6. **Analyzes** ethical implications and emerging NLP trends

> This is a real-world technique called **knowledge distillation** — training a smaller "student" model to imitate a larger "teacher" model.

---

## 2 Why This Matters

| Concern | GPT-5.2 (API) | Fine-Tuned Qwen (Self-Hosted) |
|---|---|---|
| Cost per image | $0.01–0.05 | ~$0 (infrastructure only) |
| Data privacy | Images sent to OpenAI | Data stays on your machine |
| Latency | ~5–15 seconds | ~30–180 seconds (no rate limits) |
| Vendor lock-in | Dependent on OpenAI | Full control |
| Customization | Limited to prompting | Trained on your specific task |

---

## 3 Dataset

Use publicly available Arabic handwritten text datasets. Choose **at least one** from the list below. Datasets with paragraphs/lines are preferred as they are more realistic for text recognition.

> **RECOMMENDATION:** Use **KHATT** or **MADCAT Arabic** for the most interesting results, as they contain full paragraphs of handwritten Arabic text — closest to real-world use cases.

### 3.1 Minimum Data Requirements

- At least **200 images** with corresponding text transcriptions
- Split: **80% training / 20% evaluation** (use `random_state=42` for reproducibility)
- Images should contain **handwritten Arabic text** (words, sentences, or paragraphs)

| Dataset | Content | Best For |
|---|---|---|
| **KHATT** *(recommended)* | Paragraphs, lines, full pages | OCR on paragraph-level handwritten text |
| IFN/ENIT | Word-level (Tunisian city names) | Word-level recognition |
| AHDB | Isolated characters and digits | Character recognition (simpler task) |
| MADCAT Arabic | High-quality scanned pages | Full-page OCR (used in DARPA projects) |
| QUWI | Paragraphs + writer identification | Paragraph-level OCR |
| HACDB | Characters only | Character recognition (simpler task) |
| AHTID/MW | Words (multi-writer) | Word-level recognition across writing styles |

---

## 4 Project Stages

### 4.1 Stage 1: Data Collection & GPT Labeling (The "Teacher")

Since your open datasets may have ground-truth transcriptions of varying quality, use GPT-5.2 as the "teacher" to create high-quality, consistent transcriptions:

1. **Load images** from your chosen dataset(s)
2. **Send each image to GPT-5.2** via the OpenAI API with a prompt like:

```
You are an Arabic handwriting recognition system. Read the handwritten
Arabic text in this image and transcribe it exactly. Return your response
as JSON:
{
  "transcription": "the Arabic text you read",
  "confidence": "high/medium/low",
  "notes": "any issues with legibility"
}
```

3. **Save the results** as JSONL files: each line contains the image reference and GPT's response
4. **Compare GPT's output with ground-truth** where available (quality check)

> **TIP:** GPT API calls for ~200 images should cost less than $5–10. Use the batch API for lower cost.

---

### 4.2 Stage 2: Data Transformation (GPT Format → Qwen Format)

GPT and Qwen use different conversation formats. Your `transform_data.py` script should:

- Convert any PDF/document inputs to PNG images (use PyMuPDF/fitz at **100 DPI**)
- Convert GPT-style chat format to Qwen2.5-VL format
- Simplify the output to essential fields only (e.g., just `transcription` and `confidence`)
- Split into train/eval sets (80/20)

> **Why simplify?** GPT's output contains many fields. A 7B model learns better when the target output is concise.

**Image handling:**

```python
# Qwen uses a dynamic resolution system
MIN_PIXELS = 4 * 28 * 28    # 3,136 pixels minimum
MAX_PIXELS = 128 * 28 * 28  # 100,352 pixels maximum

# DPI settings for PDF-to-image conversion
DPI = 100  # Good balance: 72 is blurry for handwriting, 200 causes OOM
```

---

### 4.3 Stage 3: Fine-Tuning with LoRA (The "Student")

Fine-tune Qwen2.5-VL-7B-Instruct using **LoRA (Low-Rank Adaptation)**.

#### 4.3.1 What is LoRA?

Instead of modifying all 7 billion parameters (which requires 80+ GB GPU memory), LoRA:

1. **Freezes** the original model weights
2. **Adds** small trainable matrices ("adapters") to specific layers
3. **Trains only** these small matrices (~2% of total parameters)

#### 4.3.2 Recommended Configuration

```python
# LoRA Configuration
LORA_CONFIG = {
    "r": 32,              # Rank — bottleneck dimension
    "lora_alpha": 64,     # Scaling factor (usually 2x rank)
    "lora_dropout": 0.05, # Regularization
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # Feed-forward layers
    ],
}

# Training Arguments
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch size = 2
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,    # Trade compute for memory
)
```

#### 4.3.3 Model Loading with 4-bit Quantization

The full model requires ~14GB in float16. With 4-bit quantization, it fits in ~4GB:

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### 4.3.4 Memory Optimizations (for 24GB GPU)

| Technique | Effect |
|---|---|
| 4-bit quantization | Base model: 14GB → ~4GB |
| Gradient checkpointing | Saves ~40% memory |
| Batch size 1 + gradient accumulation 2 | One sample at a time |
| Image pixel limits | Caps vision tokens per image |

#### 4.3.5 Key Hyperparameter: MAX_SEQ_LENGTH

```python
# IMPORTANT: Set this high enough or ALL samples get filtered out!
MAX_SEQ_LENGTH = 8192  # Vision tokens from images can easily exceed 4096
```

---

### 4.4 Stage 4: Evaluation

Evaluate your fine-tuned model against GPT's transcriptions on the held-out eval set.

| Metric | Description |
|---|---|
| **CER** (Character Error Rate) | Edit distance at character level between Qwen and GPT outputs |
| **WER** (Word Error Rate) | Edit distance at word level |
| **Exact Match Rate** | Percentage of samples where Qwen output == GPT output exactly |
| **Valid JSON Rate** | Percentage of outputs that are parseable JSON |
| **Average Inference Time** | Seconds per image on your hardware |

#### 4.4.1 Handling Invalid Outputs

The model may not always produce valid JSON. Build a robust `parse_output()` function:

1. **Direct parse:** Try `json.loads()` on the raw output
2. **Strip markdown:** Remove code block markers
3. **Truncation search:** Try every `}` from the end to find valid JSON
4. **Auto-completion:** Try adding `]}`, `"}`, etc. to close truncated output

#### 4.4.2 JSON Prefix Forcing (Important!)

Qwen tends to output `<tool_call>` or `<|im_end|>` instead of JSON. Force the model to start with a JSON prefix:

```python
json_prefix = '{"transcription":"'
text += json_prefix  # Append after the generation prompt

# Also suppress unwanted tokens
suppress = ["<tool_call>", "<|tool_call|>", "<tool_response>", "<|tool_response|>"]
gen_kwargs["bad_words_ids"] = get_token_ids(suppress)
```

---

### 4.5 Stage 5: Inference Server Deployment

Deploy your fine-tuned model using **vLLM**:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules "arabic-ocr=/path/to/lora_adapter" \
    --served-model-name "arabic-ocr-v1" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 --dtype auto
```

> **IMPORTANT:** `--max-lora-rank` must match your adapter's rank (32). vLLM defaults to 16, which will crash.

---

### 4.6 Stage 6: Downstream NLP Analysis

Apply **classical NLP techniques** to the Arabic text produced by your model. You must implement **at least two** of the following:

#### 4.6.1 Named Entity Recognition (NER)

Apply Arabic NER to the transcribed text using a pre-trained model (e.g., CAMeL Tools or AraBERT-NER). Compare NER results on:
- Ground-truth text
- GPT's transcription
- Your Qwen transcription

Report precision, recall, and F1-score.

#### 4.6.2 POS Tagging

Apply Part-of-Speech tagging using a pre-trained Arabic POS tagger (e.g., CAMeL Tools, Stanza, or Farasa). Compare POS tag sequences across the three text sources. Report tag-level accuracy.

#### 4.6.3 Sentiment Analysis *(if applicable)*

If your dataset contains opinionated or expressive text, apply Arabic sentiment analysis using a pre-trained model (e.g., AraBERT fine-tuned for sentiment). Report whether sentiment labels are preserved across transcription methods.

---

### 4.7 Stage 7: Baseline Comparison

Implement **at least one** baseline model and compare its performance against your fine-tuned Qwen:

| Baseline | Description |
|---|---|
| **CNN + BiLSTM + CTC** | The standard pre-transformer HTR architecture |
| **TrOCR** | A pre-trained transformer-based OCR model, fine-tuned on your Arabic dataset |
| **AraBERT text correction** | Feed Qwen's output through a BERT-based Arabic model to post-correct errors |

Report CER and WER for each baseline under the same evaluation protocol.

---

### 4.8 Stage 8: Ethical and Trend Analysis

Write a dedicated section (1–2 pages in your report) addressing:

#### 4.8.1 Emerging NLP Trends

- The shift from task-specific models to general-purpose VLMs for document understanding
- Knowledge distillation as a strategy for democratizing access to AI capabilities
- The role of LoRA and parameter-efficient fine-tuning in making LLMs accessible
- Multilingual and low-resource language processing (Arabic as a case study)

#### 4.8.2 Ethical Considerations

| Topic | Questions to Address |
|---|---|
| **Data privacy** | Handwritten documents may contain personal information. Discuss data handling and anonymization. |
| **Bias and fairness** | Does the model perform equally across different Arabic dialects, handwriting styles, genders, or age groups? |
| **Intellectual property** | Implications of using a commercial model (GPT) as a teacher — who owns the distilled knowledge? |
| **Dual use** | Handwriting recognition could be used for surveillance or forgery. Discuss responsible deployment. |
| **Environmental impact** | Report the compute cost (GPU hours, CO2 estimate) of your training pipeline. |

---

## 5 Compute Resources

### Option A: Google Colab (Free/Pro)

- Colab Pro provides A100 (40GB) or L4 (24GB) — sufficient for training
- Free tier T4 (16GB) — may work with aggressive quantization, but tight

### Option B: AWS/GCP/Azure

- **Recommended:** `g5.xlarge` (NVIDIA A10G, 24GB) — ~$1/hr on-demand
- Training takes ~2–4 hours depending on dataset size
- Total cost: ~$3–8 per training run
- See the *AWS Setup Guide* document for step-by-step instructions

### Option C: University Lab GPUs

- Any NVIDIA GPU with 24GB+ VRAM (A10G, A100, RTX 3090/4090)

---

## 6 Deliverables

### 6.1 Code (GitHub Repository)

```
project/
├── config.py           # All hyperparameters
├── collect_data.py     # Stage 1: Send images to GPT, save JSONL
├── transform_data.py   # Stage 2: Convert GPT format to Qwen format
├── train.py            # Stage 3: LoRA fine-tuning
├── evaluate.py         # Stage 4: Evaluation metrics
├── downstream_nlp.py   # Stage 6: NER, POS tagging
├── baseline.py         # Stage 7: Classical model baseline
├── demo.py             # Stage 5: Simple inference demo
├── requirements.txt    # Dependencies
└── README.md           # Setup and usage instructions
```

### 6.2 Report (Completed Research Paper)

Your project report is the provided `paper.tex` template, which you must complete as a full research paper. The Introduction and Related Work sections (with 34 citations) are already written for you. You must complete the remaining sections marked with orange "YOUR TASK" boxes:

1. **Proposed Methodology** (Stages 1–8): Describe your pipeline, architectural choices, hyperparameters, downstream NLP setup, baseline model, and ethical considerations
2. **Experiments:** Fill in all tables with your results — dataset statistics, training metrics, recognition performance, NER/POS results, baseline comparison, and cost analysis
3. **Results & Discussion:** Thorough analysis covering all 11 discussion points (teacher–student gap, error analysis, downstream NLP impact, baseline comparison, ethical analysis, etc.)
4. **Challenges Encountered:** Document every technical issue you faced with root cause and fix
5. **Conclusion:** Summarize findings and suggest future work

> **IMPORTANT:** The paper *is* your report — there is no separate report to write. Submit the completed `paper.pdf` as your project report. It will be graded both on technical content (experiments, results) and on writing quality (clarity, structure, proper use of citations, academic tone).

### 6.3 Demo (5-minute Presentation)

- Live demo of your inference server reading Arabic handwriting
- Show a side-by-side comparison: GPT vs. your fine-tuned Qwen
- Present NER/POS results on the transcribed text

---

## 7 Common Pitfalls & Tips

| Problem | Cause | Fix |
|---|---|---|
| OOM during training | LoRA rank too high or images too large | Reduce `r` from 64→32, reduce `MAX_PIXELS` |
| 0 training samples | `MAX_SEQ_LENGTH` too low | Increase to 8192; print token counts before training |
| Model outputs only `<\|im_end\|>` | Model learned to stop immediately | Use JSON prefix forcing during inference |
| Unparseable JSON output | Model mixes input text into output | Simplify target format, build robust parser |
| `python: command not found` | System has `python3` not `python` | Always use `python3` and `pip3` |
| LoRA rank mismatch at inference | vLLM default max rank = 16 | Set `--max-lora-rank 32` |
| Blurry handwriting | DPI too low | Use 100 DPI minimum for handwriting |
| Train/inference mismatch | Different DPI or pixel bounds | Use identical image settings in both stages |

---

## 8 Grading Rubric

### 8.1 CLO Alignment

| CLO | Description | PLO |
|---|---|---|
| CLO-2 | Implement NLP techniques including NER, POS tagging, and sentiment analysis on practical datasets | PLO 2 |
| CLO-3 | Use pre-trained models such as RNNs, LSTMs, sequence-to-sequence models, BERT, and LLMs | PLO 2, 6 |
| CLO-4 | Analyze emerging NLP trends and apply technical and ethical insights to real-world projects | PLO 1, 6 |

### 8.2 Component Breakdown

| Component | Weight | CLO-2 | CLO-3 | CLO-4 |
|---|---|---|---|---|
| Data Pipeline (Stages 1–2) | 10% | | ✓ | |
| LoRA Fine-Tuning (Stage 3) | 10% | | ✓ | |
| Evaluation (Stage 4) | 10% | | ✓ | |
| Downstream NLP — NER/POS (Stage 6) | 10% | ✓ | ✓ | |
| Baseline Comparison (Stage 7) | 10% | | ✓ | |
| Ethics & Trend Analysis (Stage 8) | 5% | | | ✓ |
| Paper: Content & Analysis | 10% | ✓ | ✓ | ✓ |
| Paper: Research & Evidence | 5% | | | ✓ |
| Paper: Writing Quality & Structure | 5% | | | ✓ |
| Paper: Challenges Documentation | 5% | | ✓ | |
| Demo & Presentation | 10% | ✓ | ✓ | ✓ |
| **Total** | **100%** | | | |

### 8.3 Detailed Rubric

| Component | Excellent (90–100%) | Good (70–89%) | Below (0–69%) |
|---|---|---|---|
| **Data Pipeline** (10%, CLO-2) | Robust GPT labeling with error handling; clean format conversion; proper train/eval split with analysis | Working pipeline with minor issues; basic format conversion | Pipeline incomplete or produces corrupt data |
| **Fine-Tuning** (15%, CLO-3) | Successful training; well-justified hyperparameters; loss curves and memory analysis reported | Training completes; default hyperparameters used; basic metrics reported | Training fails or no documentation of process |
| **Evaluation** (10%, CLO-3) | CER, WER, exact match, JSON validity all reported; robust parser implemented; qualitative error analysis | Basic metrics reported; some analysis | Metrics missing or incorrectly computed |
| **Downstream NLP** (15%, CLO-2,3) | 2+ NLP tasks implemented; comparison across GT/GPT/Qwen; error propagation analysis | 1 NLP task implemented with basic comparison | No downstream NLP analysis |
| **Baseline** (10%, CLO-3) | Classical model implemented and compared fairly; trade-off discussion | Baseline attempted with partial results | No baseline comparison |
| **Ethics & Trends** (5%, CLO-4) | Thoughtful, specific analysis of 4+ ethical dimensions; connects to current NLP trends with citations | Covers 2–3 ethical points; mentions trends | Superficial or missing ethical analysis |
| **Paper: Content & Analysis** (10%, CLO-2,3,4) | All "YOUR TASK" sections completed thoroughly; results tables filled with meaningful data; insightful discussion of all 11 points | Most sections completed; tables filled; discussion covers main points | Major sections incomplete; tables empty or fabricated |
| **Paper: Research & Evidence** (5%, CLO-4) | Existing citations used appropriately; additional references added where relevant; claims supported by data | Existing citations preserved; some claims supported | Citations removed or misused; unsupported claims |
| **Paper: Writing & Structure** (5%, CLO-4) | Clear academic writing; professional tone; logical flow; correct grammar; figures and tables well-captioned | Readable; mostly professional; minor structural issues | Unclear writing; informal tone; disorganized; many errors |
| **Paper: Challenges** (5%, CLO-3) | Honest documentation of 3+ challenges with error messages, root causes, and fixes; lessons learned | Documents 1–2 challenges with fixes | No challenges documented or fabricated |
| **Demo** (10%, CLO-2,3,4) | Live working demo; clear explanation; side-by-side GPT vs. Qwen comparison; handles Q&A well | Demo works with some issues; adequate explanation | Demo fails or no presentation |

---

## 9 Bonus Challenges (Extra Credit)

| Challenge | Bonus |
|---|---|
| Compare LoRA rank values (`r=8, 16, 32, 64`) and report the trade-offs | +5% |
| Evaluate against the dataset's ground-truth transcriptions (not just GPT), creating a 3-way comparison | +5% |
| Implement constrained decoding (e.g., using the `outlines` library) to guarantee valid JSON output | +5% |
| Fine-tune a second, smaller model (Qwen2.5-VL-3B) and compare performance vs. the 7B | +5% |
| Add support for mixed Arabic-English handwritten text | +5% |

---

## 10 Required Libraries

```txt
torch>=2.0
transformers>=4.40
peft>=0.10
bitsandbytes>=0.43
accelerate>=0.30
vllm>=0.4
Pillow
PyMuPDF       # fitz, for PDF-to-image conversion
openai        # For GPT API calls
camel-tools   # For Arabic NER and POS tagging
```

---

## 11 Getting Started Checklist

- [ ] Form your team (2–3 members)
- [ ] Choose your dataset(s) and download them
- [ ] Get an OpenAI API key (you will need ~$5–10 of credits)
- [ ] Set up your compute environment (Colab Pro recommended)
- [ ] Run GPT on a small batch (10 images) to verify your data pipeline works
- [ ] Scale up to your full dataset
- [ ] Fine-tune and iterate
- [ ] Run downstream NLP tasks on the transcribed text
- [ ] Implement at least one baseline model
- [ ] Evaluate, write your report, and prepare your demo

Good luck!
