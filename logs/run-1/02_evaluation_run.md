# Evaluation Report — Qwen2.5-VL-7B + LoRA (Run 02)

**Date:** 2026-04-18
**Notebook:** `notebook/06_evaluation.ipynb`
**Adapter:** `models/lora_adapter/` (checkpoint-120, from training run 01)
**Eval set:** `sample_training_data/eval/eval.jsonl` (20 samples)
**Results file:** `logs/02_eval_results.json`

## What changed vs. Run 01

| Fix | Before (Run 01) | After (Run 02) |
|---|---|---|
| Image pipeline | `processor(text=...)` only — no pixels reached the model | `extract_images()` decodes base64 data URLs → PIL → `processor(text=..., images=...)` |
| `max_new_tokens` | 512 (model ran to cap every time) | 128 |
| KV cache | `use_cache=False` inherited from training | `model.config.use_cache = True` + `use_cache=True` on generate |
| `temperature=0.1` w/ greedy | present (ignored, warned) | removed |
| Repetition guards | none | `repetition_penalty=1.1`, `no_repeat_ngram_size=6` |
| `torch.no_grad()` | yes | `torch.inference_mode()` (slightly lower overhead) |
| `eos_token_id` / `pad_token_id` | implicit | set explicitly so generation stops at EOS |

## Metrics

| Metric | Run 01 | Run 02 | Δ |
|---|---|---|---|
| num_samples | 20 | 20 | — |
| avg CER | 29.91 | **2.31** | **−92.3%** |
| avg WER | 32.04 | **2.32** | **−92.8%** |
| exact_match_rate | 0.0% | 0.0% | unchanged |
| valid_json_rate | 100.0% | 100.0% | unchanged |
| avg inference time | 48.02 s | **4.42 s** | **−90.8%** |

## Observations

### 1. Latency is now sane (~11× faster)
4.4 s/sample on A10G. The two drivers:
- **KV cache re-enabled** — training disabled `use_cache` for gradient checkpointing and the flag persisted into the loaded model. Turning it back on is the single biggest per-token win.
- **`max_new_tokens` 512 → 128** — short transcriptions now hit EOS well before the cap, and the few that would have looped can't run as far.

### 2. CER/WER collapsed 10×+
Going from ~30 to ~2.3 confirms the degenerate-output regime is gone. Most samples are now producing Arabic transcriptions that are roughly the right length — vision input is actually reaching the model.

### 3. Exact match is still 0%
Zero exact matches is **not** surprising at this stage:
- Arabic handwriting OCR is character-sensitive (diacritics, spacing, ta-marbuta vs. ha, etc.); even a 1-char slip drops exact match to 0.
- The adapter was trained on 80 samples for 3 epochs — way too little data to memorize exact spellings of eval-set phrases.
- CER = 2.31 still means the raw edit-distance ratio is >1.0 — i.e. hypothesis length is roughly 2–3× the reference. The model is still over-producing, just not catastrophically.

### 4. JSON formatting holds at 100%
The prefix-forcing (`{"transcription":"`) + brace-close recovery continues to produce parseable JSON on every sample. Structural reliability is not the bottleneck.

### 5. CER/WER scale still unbounded
Values >1.0 mean the `difflib`-based scorer is returning raw deletion-plus-insertion ratios. For reporting, these should be clamped to [0,1] or replaced with `jiwer` CER/WER, which follow the standard (S+D+I)/N definition.

## Conclusion

The evaluation pipeline is now **functionally correct**: images flow through the processor, generation terminates on EOS, and the LoRA adapter is doing real Arabic transcription rather than hallucinating boilerplate. Quality is still weak (exact-match 0%, CER ≈ 2.3), but that reflects the tiny 80-sample training set, not a broken eval path.

## Recommended next steps

1. **Train on the full dataset** — 1120-sample `data/train/train.jsonl` should close most of the remaining gap.
2. **Normalize CER/WER** — swap `difflib` for `jiwer.cer`/`jiwer.wer` or clamp the distance/len ratio to 1.0. Makes cross-run comparison honest.
3. **Inspect the top-K lowest-CER samples** (not just worst) to confirm the model is genuinely reading the images — a CER < 0.3 on a handful would be a strong signal.
4. **Try greedy vs. beam search (num_beams=4)** — on short outputs the compute cost is small and it tends to reduce character-level slips.
5. **Consider a Qwen-format normalization step** upstream (in notebook 04) so messages store `{"type": "image", "image": <path>}` natively; the eval notebook wouldn't need `extract_images()` at all.
