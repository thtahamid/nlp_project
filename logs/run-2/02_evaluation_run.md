# Evaluation Report — Qwen2.5-VL-7B + LoRA (Run 02)

**Date:** 2026-04-18
**Notebook:** `notebook/06_evaluation.ipynb`
**Adapter:** `models/lora_adapter/run-2/` (merged final = `checkpoint-1680`, end of epoch 3)
**Eval set:** `data/eval/eval.jsonl` (280 samples, full held-out split)
**Results file:** `logs/run-2/eval_results.json`
**Reference labels:** `gpt-5.4-mini` (see `notebook/03_data_collection.ipynb`)

## Setup

- **Base model:** `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4 + double quant, bf16 compute)
- **Adapter load:** `PeftModel.from_pretrained(base, 'models/lora_adapter/run-2/')` — confirmed via `model.peft_config` (r=32, α=64, dropout=0.05, targets q/k/v/o + gate/up/down_proj)
- **Image sizing:** `MIN_PIXELS=3,136` / `MAX_PIXELS=100,352` (matches training)
- **Scorer:** `jiwer.cer` / `jiwer.wer` (replaces the Run 01 `difflib` ratio)
- **Generation:** greedy (`do_sample=False`), `max_new_tokens=128`, KV cache on, `repetition_penalty=1.1`, `no_repeat_ngram_size=6`, JSON-prefix forcing (`{"transcription":"`), `bad_words_ids` suppressing `<tool_call>` tokens
- **Hardware:** NVIDIA A10G (23 GB)

## What changed vs. Run 01 eval

| | Run 01 eval (post-fix) | Run 02 eval |
|---|---|---|
| Eval set size | 20 (`sample_training_data/eval/`) | **280** (`data/eval/`, full) |
| Adapter source | 80-sample training, 120 steps | **1120-sample training, 1680 steps** |
| Scorer | `difflib` ratio (unbounded) | **`jiwer`** (standard `(S+D+I)/N`) |
| Output file | `logs/02_eval_results.json` | `logs/run-2/eval_results.json` |

## Metrics

| Metric | Run 01 eval (post-fix) | **Run 02** | Δ |
|---|---|---|---|
| num_samples | 20 | 280 | 14× more |
| avg CER | 2.31 (difflib) | **1.4505** (jiwer) | lower, but scorers differ |
| avg WER | 2.32 (difflib) | **1.5765** (jiwer) | — |
| exact_match_rate | 0.0% | **0.0%** | unchanged |
| valid_json_rate | 100.0% | **100.0%** | unchanged |
| avg inference time | 4.42 s | **3.52 s** | −20% |

> **CER / WER can exceed 1.0.** `jiwer` computes `(Substitutions + Deletions + Insertions) / N_reference`, so insertions push the ratio above 1 when the hypothesis is longer than the reference. A CER of 1.45 means edit operations outnumber reference characters.

## Observations

### 1. Exact-match is still 0% — and worse, the model has a canned fallback

The five worst samples all produce the **identical** 17-word sentence:

> الصينية، وهم يدعون إلى التفاوض على تسوية النزاعات من خلال الحوار والتفاهم المتبادل.

(*"…Chinese, calling for negotiating to resolve disputes through dialogue and mutual understanding."*)

Against short single-word references (جميلة, تأشيرة, الصنيون, فنكتمكنه, ان يقبسه) that comparison yields CER 9.6–15.8. This is the same failure mode as Run 01 (the "smoking" loop), just capped earlier by `max_new_tokens=128` and shorter overall.

**Confirmed the phrase is not a training-data artefact.** A grep of `data/train/train.jsonl` for the fallback phrase returns **0 / 1120 matches** — it is not memorised from the adapter's training set. It is a pretraining prior from the base Qwen2.5-VL model that leaks through whenever the vision signal is too weak to steer generation.

### 2. Short references dominate the failure tail

Eval reference lengths: `min=5, median=61, mean=60, max=128`. Only **8 / 280 (2.9%)** references are ≤10 chars, yet every one of the top-5 highest-CER samples is in that tail. The CER denominator is tiny, so any over-production explodes the ratio. The fallback sentence is ~85 chars — when the reference is 5 chars, `(S+D+I)/N ≈ 80/5 = 16`.

The averaged CER of 1.45 is therefore a mix of (a) reasonable errors on medium/long references and (b) a handful of catastrophic misses on short ones. The aggregate understates how well the model does on longer handwriting and overstates the typical error.

### 3. JSON formatting is perfect

`valid_json_rate = 100%` — prefix forcing + brace-close recovery continues to deliver parseable output on every sample. Structural reliability is not a bottleneck.

### 4. Latency is healthy

3.52 s/sample averaged across 280 samples — 20% faster than Run 01's 4.42 s despite 14× more samples. KV cache + `max_new_tokens=128` ensure generation usually terminates on EOS before the token cap. No sample hung.

### 5. Checkpoint choice matters — and we may have picked the wrong one

Training (`01_training_run.md`) showed eval loss `1.6350 → 1.6248 → 1.6387` across epochs 1/2/3, with train/eval gap widening in epoch 3 (train 1.46 vs eval 1.64). This evaluation used the **merged final checkpoint (1680, epoch 3)**. The epoch-2 checkpoint (`checkpoint-1120`) had the lowest eval loss and is the better candidate. A comparison run is queued.

## Conclusion

The pipeline is correct and fast, but **Run 02 has not solved the problem**. The adapter successfully learned JSON structure and rough Arabic token distributions, yet for a non-trivial fraction of samples the vision signal still doesn't override the base model's pretraining priors — producing a canned, topical fallback instead of reading the handwriting. Training loss at 1.45 is still high, the fallback is a base-model prior not present in our training data, and the reliance on a single merged checkpoint from an over-trained epoch likely hurts further.

## Recommended next steps

1. **Evaluate `checkpoint-1120`.** Set `CHECKPOINT = 'checkpoint-1120'` in §4 of the eval notebook and rerun §5→§8. Writes `eval_results_checkpoint-1120.json`. Expected: lower CER/WER, since epoch-3 overfits on 1120 training samples.
2. **Stratify metrics by reference length.** Report CER/WER separately for short (`N ≤ 10`), medium (`10 < N ≤ 60`), and long (`> 60`) references. The headline average is dominated by a handful of short-ref blow-ups and masks real performance.
3. **Clamp CER/WER for headline reporting.** Many OCR papers clamp to `min(CER, 1.0)` so a single runaway sample can't double the mean.
4. **Investigate why vision signal fails on short samples.** Likely a training distribution problem — if training refs skew long, the model never learned to *stop* early. Check `len(ref)` distribution in `train.jsonl` vs `eval.jsonl`.
5. **Train longer or with more data** if both checkpoints disappoint. Final train loss 1.46 is still well above the regime where transcription gets reliably correct; consider epoch-4/5 with weight decay, or augment the dataset.
6. **Optional: constrain generation harder.** Experiment with `max_new_tokens=64`, a stronger `repetition_penalty`, or a logit processor that forces closing `"}` after N tokens for short-ref cases.
