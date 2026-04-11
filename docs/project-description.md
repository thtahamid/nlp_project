# NLP Final Project

## Knowledge Distillation from GPT to Qwen

### for Arabic Handwritten Text Recognition

MAI 656 — Natural Language Processing  
Canadian University Dubai  
Spring 2026

- Course: MAI 656 — Natural Language Processing
- Due Date: TBD
- Team Size: 2–3 students
- Weight: 35% (Report 20% + Presentation 10% + Code 5%)

## Contents

- 1 Project Overview
- 1.1 Your Goal
- 2 Why This Matters
- 3 Dataset
- 3.1 Minimum Data Requirements
- 4 Project Stages
- 4.1 Stage 1: Data Collection & GPT Labeling (The “Teacher”)
- 4.2 Stage 2: Data Transformation (GPT Format→Qwen Format)
- 4.3 Stage 3: Fine-Tuning with LoRA (The “Student”)
- 4.3.1 What is LoRA?
- 4.3.2 Recommended Configuration
- 4.3.3 Model Loading with 4-bit Quantization
- 4.3.4 Memory Optimizations (for 24GB GPU)
- 4.3.5 Key Hyperparameter: MAX_SEQ_LENGTH
- 4.4 Stage 4: Evaluation
- 4.4.1 Handling Invalid Outputs
- 4.4.2 JSON Prefix Forcing (Important!)
- 4.5 Stage 5: Inference Server Deployment
- 4.6 Stage 6: Downstream NLP Analysis
- 4.6.1 Named Entity Recognition (NER)
- 4.6.2 POS Tagging
- 4.6.3 Sentiment Analysis (if applicable)
- 4.7 Stage 7: Baseline Comparison
- 4.8 Stage 8: Ethical and Trend Analysis
- 4.8.1 Emerging NLP Trends
- 4.8.2 Ethical Considerations

## 1 Project Overview

Large commercial models like OpenAI’s GPT-5.2 can read handwritten Arabic text from images
with high accuracy — but they are expensive, slow, and require sending data to external servers.
In this project, you willdistillthat capability into a smaller, open-source model (Qwen2.5-
VL-7B) that can run locally on a single GPU.

### 1.1 Your Goal

Build an end-to-end pipeline that:

1. Uses GPT-5.2 to generate high-quality transcriptions of Arabic handwritten text images
   (creating your training data)
2. Fine-tunes Qwen2.5-VL-7B-Instruct using LoRA to replicate GPT’s transcription ability
3. Applies downstream NLP tasks (NER, POS tagging) to the transcribed Arabic text
4. Comparesthedistilledmodelagainstclassicalbaselines(CNN+BiLSTM+CTCorTrOCR)
5. Evaluates how closely your fine-tuned model matches GPT’s output
6. Analyzes ethical implications and emerging NLP trends
   This is a real-world technique calledknowledge distillation— training a smaller “student”
   model to imitate a larger “teacher” model.

## 2 Why This Matters

Table 1: GPT-5.2 API vs. a fine-tuned self-hosted Qwen model.
Concern GPT-5.2 (API) Fine-Tuned Qwen (Self-Hosted)
Cost per image $0.01–0.05 ~$0 (infrastructure only)
Data privacy Images sent to OpenAI Data stays on your machine
Latency ~5–15 seconds ~30–180 seconds (no rate limits)
Vendor lock-in Dependent on OpenAI Full control
Customization Limited to prompting Trained on your specific task

## 3 Dataset

You will use publicly available Arabic handwritten text datasets. Chooseat least onefrom
the list below. Datasets with paragraphs/lines are preferred as they are more realistic for text
recognition.
RECOMMENDATION
UseKHATTorMADCAT Arabicfor the most interesting results, as they contain
full paragraphs of handwritten Arabic text — closest to real-world use cases.

### 3.1 Minimum Data Requirements

- At least200 imageswith corresponding text transcriptions
- Split:80% training / 20% evaluation(userandom_state=42for reproducibility)
- Images should containhandwritten Arabic text(words, sentences, or paragraphs)

Table 2: Recommended Arabic handwriting datasets.
Dataset Content Best For
KHATTParagraphs, lines, full pages OCR on paragraph-level handwritten text
(recommended)
IFN/ENITWord-level (Tunisian city names) Word-level recognition
AHDBIsolated characters and digits Character recognition (simpler task)
MADCAT ArabicHigh-quality scanned pages Full-page OCR (used in DARPA projects)
QUWIParagraphs + writer identification Paragraph-level OCR
HACDBCharacters only Character recognition (simpler task)
AHTID/MWWords (multi-writer) Word-level recognition across writing
styles

## 4 Project Stages

### 4.1 Stage 1: Data Collection & GPT Labeling (The “Teacher”)

Since your open datasets may have ground-truth transcriptions of varying quality, you will use
GPT-5.2 as the “teacher” to create high-quality, consistent transcriptions:

1. Load imagesfrom your chosen dataset(s)
2. Send each image to GPT-5.2via the OpenAI API with a prompt like:
   You are an Arabic handwriting recognition system. Read the handwritten
   Arabic text in this image and transcribe it exactly. Return your
   response
   as JSON:
   {
   "transcription": "the Arabic text you read",
   "confidence": "high/medium/low",
   "notes": "any issues with legibility"
   }
3. Save the resultsas JSONL files: each line contains the image reference and GPT’s
   response
4. Compare GPT’s output with ground-truthwhere available (this gives you a quality
   check)
   TIP
   GPT API calls for ~200 images should cost less than $5–10. Use the batch API for lower
   cost.

### 4.2 Stage 2: Data Transformation (GPT Format→Qwen Format)

GPT and Qwen use different conversation formats. Yourtransform_data.pyscript should:

- Convert any PDF/document inputs to PNG images (use PyMuPDF/fitz at100 DPI)
- Convert GPT-style chat format to Qwen2.5-VL format
- Simplify the outputto essential fields only (e.g., justtranscriptionandconfidence)
- Split into train/eval sets (80/20)

Why simplify?GPT’s output contains many fields. A 7B model learns better when the
target output is concise.
Image handling:

# Qwen uses a dynamic r e s o l u t i o n system

MIN_PIXELS = 4 _ 28 _ 28# 3 ,136 pixels minimum
MAX_PIXELS = 128 _ 28 _ 28# 100 ,352 pixels maximum

# DPI settings for PDF - to - image c o n v e r s i o n

DPI = 100# Good balance : 72 is blurry for handwriting , 200 causes
OOM

### 4.3 Stage 3: Fine-Tuning with LoRA (The “Student”)

Fine-tune Qwen2.5-VL-7B-Instruct usingLoRA (Low-Rank Adaptation).

#### 4.3.1 What is LoRA?

Instead of modifying all 7 billion parameters (which requires 80+ GB GPU memory), LoRA:

1. Freezesthe original model weights
2. Addssmall trainable matrices (“adapters”) to specific layers
3. Trains onlythese small matrices (~2% of total parameters)

#### 4.3.2 Recommended Configuration

# LoRA C o n f i g u r a t i o n

LoRA_CONFIG = {
"r": 32,# Rank -- b o t t l e n e c k d im en sio n
"lora_alpha": 64,# Scaling factor ( usually 2 x rank )
"lora_dropout": 0.05,# R e g u l a r i z a t i o n
"target_modules": [
"q_proj", "k_proj", "v_proj", "o_proj",# A tt en tio n layers
"gate_proj", "up_proj", "down_proj",# Feed - forward
layers
],
}

# Training A rg um ent s

TrainingArguments(
num_train_epochs=3,
per_device_train_batch_size=1,
gradient_accumulation_steps=2,# E ffe ct iv e batch size = 2
learning_rate=2e-5,
lr_scheduler_type="cosine",
warmup_ratio=0.1,
bf16=True,
gradient_checkpointing=True,# Trade compute for memory
)

#### 4.3.3 Model Loading with 4-bit Quantization

The full model requires ~14GB in float16. With 4-bit quantization, it fits in ~4GB:

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

#### 4.3.4 Memory Optimizations (for 24GB GPU)

1. 4-bit quantization: Base model from 14GB→~4GB
2. Gradient checkpointing: Saves ~40% memory
3. Batch size 1 + gradient accumulation 2: One sample at a time
4. Image pixel limits: Caps vision tokens per image

#### 4.3.5 Key Hyperparameter: MAX_SEQ_LENGTH

# IM PO RT ANT : Set this high enough or ALL samples get filtered out !

MAX_SEQ_LENGTH = 8192# Vision tokens from images can easily exceed
4096

### 4.4 Stage 4: Evaluation

Evaluate your fine-tuned model against GPT’s transcriptions on the held-out eval set.
Metrics to report:
Metric Description
CER(Character Error Rate) Edit distance at character level between Qwen and
GPT outputs
WER(Word Error Rate) Edit distance at word level
Exact Match RatePercentage of samples where Qwen output == GPT
output exactly
Valid JSON RatePercentage of outputs that are parseable JSON
Average Inference TimeSeconds per image on your hardware

#### 4.4.1 Handling Invalid Outputs

The model may not always produce valid JSON. Build a robustparse_output()function:

1. Direct parse: Tryjson.loads()on the raw output
2. Strip markdown: Remove code block markers
3. Truncation search: Try every}from the end to find valid JSON
4. Auto-completion: Try adding]},"]}, etc. to close truncated output

#### 4.4.2 JSON Prefix Forcing (Important!)

Qwen tends to output<tool_call>or<|im_end|>instead of JSON. Force the model to start
with a JSON prefix:
json_prefix = ’{"transcription":"’
text += json_prefix# Append after the g e n e r a t i o n prompt

# Also suppress unwanted tokens

suppress = ["<tool_call>", "<|tool_call|>",
"<tool_response>", "<|tool_response|>"]
gen_kwargs["bad_words_ids"] = get_token_ids(suppress)

### 4.5 Stage 5: Inference Server Deployment

Deploy your fine-tuned model usingvLLM:
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
IMPORTANT
–max-lora-rankmust match your adapter’s rank (32). vLLM defaults to 16, which will
crash.

### 4.6 Stage 6: Downstream NLP Analysis

ApplyclassicalNLPtechniquestotheArabictextproducedbyyourmodel. Youmustimplement
at least twoof the following:

#### 4.6.1 Named Entity Recognition (NER)

Apply Arabic NER to the transcribed text using a pre-trained model (e.g., CAMeL Tools or
AraBERT-NER). Compare NER results on ground-truth text vs. GPT’s transcription vs. your
Qwen transcription. Report precision, recall, and F1-score.

#### 4.6.2 POS Tagging

Apply Part-of-Speech tagging using a pre-trained Arabic POS tagger (e.g., CAMeL Tools,
Stanza, or Farasa). Compare POS tag sequences across the three text sources. Report tag-
level accuracy.

#### 4.6.3 Sentiment Analysis (if applicable)

If your dataset contains opinionated or expressive text, apply Arabic sentiment analysis using a
pre-trained model (e.g., AraBERT fine-tuned for sentiment). Report whether sentiment labels
are preserved across transcription methods.

### 4.7 Stage 7: Baseline Comparison

Implementat least onebaseline model and compare its performance against your fine-tuned
Qwen:

- CNN + BiLSTM + CTC:The standard pre-transformer HTR architecture
- TrOCR:A pre-trained transformer-based OCR model, fine-tuned on your Arabic dataset
- AraBERT text correction:Feed Qwen’s output through a BERT-based Arabic model
  to post-correct errors
  Report CER and WER for each baseline under the same evaluation protocol.

### 4.8 Stage 8: Ethical and Trend Analysis

Write a dedicated section (1–2 pages in your report) addressing:

#### 4.8.1 Emerging NLP Trends

- The shift from task-specific models to general-purpose VLMs for document understanding
- Knowledge distillation as a strategy for democratizing access to AI capabilities
- The role of LoRA and parameter-efficient fine-tuning in making LLMs accessible
- Multilingual and low-resource language processing (Arabic as a case study)

#### 4.8.2 Ethical Considerations

- Data privacy:Handwritten documents may contain personal information. Discuss data
  handling and anonymization.
- Bias and fairness:Does the model perform equally across different Arabic dialects,
  handwriting styles, genders, or age groups?
- Intellectual property:Implications of using a commercial model (GPT) as a teacher —
  who owns the distilled knowledge?
- Dual use:Handwriting recognition could be used for surveillance or forgery. Discuss
  responsible deployment.
- Environmental impact:Report the compute cost (GPU hours, CO2 estimate) of your
  training pipeline.

## 5 Compute Resources

### 5.1 Option A: Google Colab (Free/Pro)

- Colab Pro provides A100 (40GB) or L4 (24GB) — sufficient for training
- Free tier T4 (16GB) — may work with aggressive quantization, but tight

### 5.2 Option B: AWS/GCP/Azure

- Recommended:g5.xlarge (NVIDIA A10G, 24GB) — ~$1/hr on-demand
- Training takes ~2–4 hours depending on dataset size
- Total cost: ~$3–8 per training run
- See theAWS Setup Guidedocument for step-by-step instructions

### 5.3 Option C: University Lab GPUs

- Any NVIDIA GPU with 24GB+ VRAM (A10G, A100, RTX 3090/4090)

## 6 Deliverables

### 6.1 Code (GitHub Repository)

project/
config.py # All hyperparameters
collect_data.py # Stage 1: Send images to GPT, save JSONL
transform_data.py # Stage 2: Convert GPT format to Qwen
format
train.py # Stage 3: LoRA fine-tuning
evaluate.py # Stage 4: Evaluation metrics
downstream_nlp.py # Stage 6: NER, POS tagging
baseline.py # Stage 7: Classical model baseline
demo.py # Stage 5: Simple inference demo
requirements.txt # Dependencies
README.md # Setup and usage instructions

### 6.2 Report = Completed Research Paper

Your project report is the providedpaper.textemplate, which you must complete as a full
research paper. The Introduction and Related Work sections (with 34 citations) are already
written for you. You must complete the remaining sections marked with orange “YOUR TASK”
boxes:
1.Proposed Methodology(Stages 1–8): Describe your pipeline, architectural choices,
hyperparameters, downstream NLP setup, baseline model, and ethical considerations
2.Experiments: Fill in all tables with your results — dataset statistics, training metrics,
recognition performance, NER/POS results, baseline comparison, and cost analysis
3.Results & Discussion: Thorough analysis covering all 11 discussion points (teacher–
studentgap, erroranalysis, downstreamNLPimpact, baselinecomparison, ethicalanalysis,
etc.)
4.Challenges Encountered: Document every technical issue you faced with root cause
and fix
5.Conclusion: Summarize findings and suggest future work
IMPORTANT
The paperisyour report — there is no separate report to write. Submit the completed
paper.pdfas your project report. It will be graded both on technical content (experi-
ments, results) and on writing quality (clarity, structure, proper use of citations, academic
tone). See the grading rubric in Section 8 for the detailed paper grading criteria.

### 6.3 Demo (5-minute presentation)

- Live demo of your inference server reading Arabic handwriting
- Show a side-by-side comparison: GPT vs. your fine-tuned Qwen
- Present NER/POS results on the transcribed text

## 7 Common Pitfalls & Tips

These are real issues encountered in production knowledge distillation projects:
Table 3: Common issues and their fixes.
Problem Cause Fix
OOM during training LoRA rank too high or im-
ages too large
Reducerfrom 64→32, reduce
MAX_PIXELS
0 training samplesMAX_SEQ_LENGTHtoo low Increase to 8192; print token counts
before training
Model outputs only
<|im_end|>
Model learned to stop imme-
diately
Use JSON prefix forcing during in-
ference
Unparseable JSON
output
Model mixes input text into
output
Simplify target format, build robust
parser
python: command
not found
System haspython3not
python
Always usepython3andpip3
LoRA rank mismatch
at inference
vLLM default max rank = 16 Set–max-lora-rank 32
Blurry handwriting DPI too low Use100DPIminimumforhandwrit-
ing
Train/inference mis-
match
Different DPI or pixel bounds Use identical image settings in both
stages

## 8 Grading Rubric

### 8.1 CLO Alignment

This project is assessed against the following Course Learning Outcomes:
Table 4: Course Learning Outcomes addressed by this project.
CLO Description PLO
CLO-2 Implement NLP techniques including NER, POS tagging, and sen-
timent analysis on practical datasets
PLO 2
CLO-3 Use pre-trained models such as RNNs, LSTMs, sequence-to-
sequence models, BERT, and LLMs
PLO 2, 6
CLO-4 Analyze emerging NLP trends and apply technical and ethical in-
sights to real-world projects
PLO 1, 6

### 8.2 Component Breakdown

Note: Some components map to multiple CLOs. The paper accounts for 25% of the total grade.

### 8.3 Detailed Rubric

## 9 Bonus Challenges (Extra Credit)

- +5%: Compare LoRA rank values (r=8, 16, 32, 64) and report the trade-offs
- +5%: Evaluate against the dataset’s ground-truth transcriptions (not just GPT), creating
  a 3-way comparison
- +5%: Implement constrained decoding (e.g., using theoutlineslibrary) to guarantee
  valid JSON output
  Table 5: Grading breakdown with CLO mapping.
  Component Weight CLO-2 CLO-3 CLO-4
  Data Pipeline (Stages 1–2) 10%✓
  LoRA Fine-Tuning (Stage 3) 10%✓
  Evaluation (Stage 4) 10%✓
  Downstream NLP — NER/POS (Stage 6) 10%✓ ✓
  Baseline Comparison (Stage 7) 10%✓
  Ethics & Trend Analysis (Stage 8) 5%✓
  Paper (Report)
  Content & Analysis 10%✓ ✓ ✓
  Research & Evidence 5%✓
  Writing Quality & Structure 5%✓
  Challenges Documentation 5%✓
  Demo & Presentation 10%✓ ✓ ✓
  Total 100%
- +5%: Fine-tune a second, smaller model (Qwen2.5-VL-3B) and compare performance vs.
  the 7B
- +5%: Add support for mixed Arabic-English handwritten text

## 10 Required Libraries

torch>=2.0
transformers >=4.40
peft>=0.10
bitsandbytes >=0.43
accelerate >=0.30
vllm>=0.4
Pillow
PyMuPDF # fitz, for PDF-to-image conversion
openai # For GPT API calls
camel-tools # For Arabic NER and POS tagging

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

Table 6: Detailed grading criteria per component.
Component Excellent (90–
100%)
Good (70–89%) Below (0–69%)
Data Pipeline
(10%, CLO-2)
Robust GPT labeling
with error handling;
clean format conver-
sion; proper train/eval
split with analysis
Working pipeline with
minor issues; basic for-
mat conversion
Pipeline incomplete or
produces corrupt data
Fine-Tuning
(15%, CLO-3)
Successful training;
well-justified hyperpa-
rameters; loss curves
and memory analysis
reported
Training completes;
default hyperparame-
ters used; basic metrics
reported
Training fails or no
documentation of pro-
cess
Evaluation
(10%, CLO-3)
CER, WER, exact
match, JSON validity
all reported; robust
parser implemented;
qualitative error anal-
ysis
Basic metrics reported;
some analysis
Metrics missing or in-
correctly computed
Downstream
NLP
(15%, CLO-2,3)
2+ NLP tasks im-
plemented; compari-
son across GT/GP-
T/Qwen; error propa-
gation analysis
1 NLP task imple-
mented with basic
comparison
No downstream NLP
analysis
Baseline
(10%, CLO-3)
Classical model imple-
mented and compared
fairly; trade-off discus-
sion
Baseline attempted
with partial results
No baseline compari-
son
Ethics&Trends
(5%, CLO-4)
Thoughtful, specific
analysis of 4+ ethical
dimensions; connects
to current NLP trends
with citations
Covers 2–3 ethical
points; mentions
trends
Superficial or missing
ethical analysis

Table 7: Detailed grading criteria — Paper and Presentation.
Component Excellent (90–
100%)
Good (70–89%) Below (0–69%)
Paper: Content
& Analysis
(10%, CLO-2,3,4)
All “YOUR TASK”
sections completed
thoroughly; results
tables filled with
meaningful data; in-
sightful discussion of
all 11 points
Most sections com-
pleted; tables filled;
discussion covers main
points
Major sections incom-
plete; tables empty or
fabricated
Paper: Re-
search & Evi-
dence
(5%, CLO-4)
Existing citations used
appropriately; addi-
tional references added
where relevant; claims
supported by data
Existing citations pre-
served; some claims
supported
Citations removed or
misused; unsupported
claims
Paper: Writing
& Structure
(5%, CLO-4)
Clear academic writ-
ing; professional tone;
logical flow; correct
grammar; figures and
tables well-captioned
Readable; mostly pro-
fessional; minor struc-
tural issues
Unclear writing; infor-
mal tone; disorganized;
many errors
Paper: Chal-
lenges
(5%, CLO-3)
Honest documentation
of 3+ challenges with
error messages, root
causes, and fixes;
lessons learned
Documents 1–2 chal-
lenges with fixes
No challenges docu-
mented or fabricated
Demo
(10%, CLO-2,3,4)
Live working demo;
clear explanation;
side-by-side GPT vs.
Qwen comparison;
handles Q&A well
Demo works with some
issues; adequate expla-
nation
Demo fails or no pre-
sentation
