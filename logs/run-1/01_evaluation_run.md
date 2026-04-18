# Evaluation Report — Qwen2.5-VL-7B + LoRA (Run 01)

**Date:** 2026-04-18
**Notebook:** `notebook/06_evaluation.ipynb`
**Adapter:** `models/lora_adapter/` (checkpoint-120, from training run 01)
**Eval set:** `sample_training_data/eval/eval.jsonl` (20 samples)
**Results file:** `logs/01_eval_results.json`

## Setup
- **Base model:** Qwen/Qwen2.5-VL-7B-Instruct (4-bit NF4 + double quant, bf16 compute)
- **Adapter:** LoRA loaded via `PeftModel.from_pretrained`
- **Image sizing:** MIN_PIXELS=3,136 / MAX_PIXELS=100,352 (matches training)
- **Hardware:** NVIDIA A10G (23 GB), ~11 GB used at idle
- **Generation:** greedy (`do_sample=False`), `max_new_tokens=512`, JSON-prefix forcing (`{"transcription":"`), `bad_words_ids` suppress `<tool_call>` tokens

## Metrics

| Metric | Value |
|--------|-------|
| num_samples | 20 |
| avg CER | **29.91** |
| avg WER | **32.04** |
| exact_match_rate | **0.0%** |
| valid_json_rate | **100.0%** |
| avg inference time | 48.02 s / sample |

> Note: CER/WER values are not normalized to [0,1] — the `difflib`-based scorer returns raw edit-distance ratios that can exceed 1.0 when the prediction is much longer than the reference. A CER of ~30 means the hypothesis is roughly 30× longer than the reference.

## Observations

### 1. Model is degenerating into a repetition loop
Every one of the top-5 worst samples produced the same runaway phrase:
> "وقد أشارت إلى أن هناك تفاوتاً في التأثيرات الصحية للتدخين بين الجنسين…"
repeated until `max_new_tokens=512` is exhausted.

- The reference texts are short Arabic handwriting transcriptions (~5–10 words).
- The model generates an unrelated long paragraph about smoking, then loops.
- This is why CER/WER are so inflated — the hypothesis is hundreds of tokens over the reference.

### 2. JSON formatting is perfect
`valid_json_rate = 100%` — the JSON-prefix forcing + brace-close recovery in `parse_output()` works as intended. Structure is not the problem; content is.

### 3. Exact match = 0
Given the repetition issue, no sample matches. This is expected fallout from (1), not an independent issue.

### 4. Inference latency is high
~48 s/sample on A10G. With `max_new_tokens=512` and greedy decoding, the model is hitting the token cap on nearly every sample (corroborating the repetition observation).

### 5. Suspected root causes

1. **Images are not being passed.** Cell-14 calls `processor(text=text, return_tensors='pt')` — **no `images=` argument and no call to `process_vision_info`** from `qwen_vl_utils`. The chat template includes `<|vision_start|>…<|vision_end|>` placeholders, but without the actual pixel tensors the model sees empty vision slots and hallucinates plausible-sounding Arabic text.
2. **Training data was tiny.** Run 01 used `sample_training_data/` (80 train / 20 eval), not the full `data/train/train.jsonl` (1120 rows). Final train loss 1.59 on 80 samples does not generalize.
3. **Greedy + bf16 quantized VL model** is known to loop when input conditioning is weak (reinforces #1).

## Conclusion

The fine-tuned adapter loads and produces syntactically valid JSON, but the evaluation code is not feeding image pixels into the model, so outputs are effectively text-only hallucinations. Metrics from this run are **not a valid measure of model quality** — they measure the model's failure mode when vision input is missing.

## Recommended next steps

1. **Fix the evaluation input pipeline** — use `process_vision_info` from `qwen_vl_utils` and pass `images=image_inputs` (and `videos=`) to the processor:
   ```python
   from qwen_vl_utils import process_vision_info
   image_inputs, video_inputs = process_vision_info(input_messages)
   inputs = processor(text=text, images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors='pt').to(model.device)
   ```
2. **Re-run training on the full dataset** (`data/train/train.jsonl`, 1120 rows) — the sample set is too small for meaningful eval.
3. **Normalize CER/WER** so they stay in [0,1] (clamp or use `jiwer`).
4. **Add a repetition penalty / no_repeat_ngram_size** to generation as a guardrail.
5. Remove `temperature=0.1` (ignored under `do_sample=False`, prints a warning).
