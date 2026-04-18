# Training Analysis — Qwen2.5-VL-7B LoRA (Run 02)

**Date:** 2026-04-18
**Checkpoint:** `models/lora_adapter/run-2/checkpoint-1680` (final, end of epoch 3)
**Reference labels:** generated with `gpt-5.4-mini` (see `notebook/03_data_collection.ipynb`)

## Setup
- **Base model:** Qwen2.5-VL-7B-Instruct (4-bit NF4, bf16 compute)
- **Adapter:** LoRA (r=32, alpha=64, dropout=0.05)
- **Targets:** q/k/v/o_proj, gate/up/down_proj
- **Data:** 1120 train / 280 eval (full split, vs. 80/20 sample in Run 01)
- **Training fix vs. Run 01:** `ArabicHTRDataset.__getitem__` now base64-decodes images and passes `images=` into the processor — Run 01 trained on empty vision slots.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 2e-5 |
| lr_scheduler | cosine |
| warmup_ratio | 0.1 |
| num_train_epochs | 3 |
| per_device_train_batch_size | 1 |
| per_device_eval_batch_size | 1 |
| gradient_accumulation_steps | 2 |
| effective_batch_size | 2 |
| optimizer | adamw_torch_fused (β1=0.9, β2=0.999, ε=1e-8) |
| weight_decay | 0.0 |
| max_grad_norm | 1.0 |
| precision | bf16 |
| gradient_checkpointing | True |
| seed | 42 |
| eval_strategy | epoch |
| save_strategy | epoch (limit=2) |
| logging_steps | 10 |
| total steps | 1680 |

(Identical to Run 01; only the data scaled. See `logs/run-1/01_training_run.md` for the term/notation glossary.)

## Loss Curve

Sampled at quarter-epoch boundaries. Eval loss reported at end of each epoch.

| Step | Epoch | Train Loss | Eval Loss | Grad Norm | LR |
|------|-------|------------|-----------|-----------|----|
| 140  | 0.25  | 1.8297     | —         | 2.03      | 1.66e-05 |
| 280  | 0.50  | 1.5575     | —         | 1.76      | 1.97e-05 |
| 420  | 0.75  | 1.6992     | —         | 1.96      | 1.87e-05 |
| 560  | 1.00  | 1.6298     | **1.6350** | 2.16      | 1.69e-05 |
| 700  | 1.25  | 1.6335     | —         | 3.03      | 1.45e-05 |
| 840  | 1.50  | 1.4733     | —         | 3.20      | 1.18e-05 |
| 980  | 1.75  | 1.6501     | —         | 3.37      | 8.86e-06 |
| 1120 | 2.00  | 1.6208     | **1.6248** | 3.14      | 6.06e-06 |
| 1260 | 2.25  | 1.4773     | —         | 4.22      | 3.59e-06 |
| 1400 | 2.50  | 1.4481     | —         | 4.21      | 1.66e-06 |
| 1540 | 2.75  | 1.4703     | —         | 3.45      | 4.26e-07 |
| 1680 | 3.00  | 1.4550     | **1.6387** | 3.97      | 2.16e-11 |

## Wall-clock

| Phase | Time |
|-------|------|
| Training start (after model load) | 2026-04-18 01:06 |
| Checkpoint-1120 saved (end ep 2) | 2026-04-18 01:40 |
| Checkpoint-1680 saved (end ep 3) | 2026-04-18 01:50 |
| Adapter merged + saved | 2026-04-18 01:56 |
| **Total training** | **~44 min** (≈38 steps/min on A10G) |
| Eval pass per epoch | ~37 s (280 samples) |

## Observations
- **First epoch dominates the drop.** Train loss falls from 8.01 → 1.63 in the first 560 steps; the next two epochs only shave another ~0.18 off the train loss.
- **Eval loss plateaus then nudges up.** 1.6350 → 1.6248 → 1.6387. The minimum is at end of epoch 2 (step 1120).
- **Train/eval gap widens in epoch 3.** Train 1.46 vs eval 1.64 — the model is starting to memorize the training set.
- **Gradient norms drift up across the run** (≈2 → ≈4) while LR decays toward 0; consistent with the model fitting harder examples late in training.
- **No instability:** no spikes, no NaNs, smooth cosine decay.

## Comparison to Run 01

| Metric | Run 01 (80 samples, broken vision) | Run 02 (1120 samples, fixed) |
|--------|-----------------------------------|------------------------------|
| Final train loss | 1.59 | 1.46 |
| Final eval loss | 1.85 | 1.64 |
| Best eval loss | 1.85 (step 120) | **1.6248 (step 1120)** |
| Total steps | 120 | 1680 |
| Wall-clock | ~2 min | ~44 min |

Run 01's eval was inflated because the vision tower was fed empty slots — the model was effectively a text-only LM trying to autoregress the transcription. Run 02 actually reads the page.

## Conclusion
Training is healthy and the **best checkpoint is `checkpoint-1120` (end of epoch 2, eval loss 1.6248)**, not the final checkpoint. The epoch-3 eval bump (1.6387) plus the rising train/eval gap suggests mild overfitting in the last epoch. Recommend evaluating both `checkpoint-1120` and `checkpoint-1680` on the held-out set; if `checkpoint-1120` wins on CER/WER, drop epoch 3 next time (or add weight decay / more dropout).
