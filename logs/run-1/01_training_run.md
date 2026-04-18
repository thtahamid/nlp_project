# Training Analysis — Qwen2.5-VL-7B LoRA

**Date:** 2026-04-17
**Checkpoint:** `models/lora_adapter/checkpoint-120` (final)

## Setup
- **Base model:** Qwen2.5-VL-7B-Instruct
- **Adapter:** LoRA (r=32, alpha=64, dropout=0.05)
- **Targets:** q/k/v/o_proj, gate/up/down_proj
- **Data:** 1120 train / 280 eval

## Hyperparameters (first run)

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
| total steps | 120 |

## Term / Notation Glossary

- **LoRA (Low-Rank Adaptation):** freezes the base model and trains small low-rank matrices injected into attention/MLP layers.
- **r (rank):** size of the LoRA bottleneck; bigger r = more capacity, more params. (r=32 here)
- **alpha:** LoRA scaling factor; effective scale = alpha/r (64/32 = 2×).
- **dropout:** randomly drops LoRA activations during training to regularize.
- **target_modules:** which base-model linear layers get LoRA adapters.
  - `q_proj, k_proj, v_proj, o_proj` = attention query/key/value/output projections.
  - `gate_proj, up_proj, down_proj` = SwiGLU MLP projections.
- **learning_rate (2e-5):** step size for the optimizer.
- **cosine scheduler:** LR rises during warmup, then decays as a cosine curve to 0.
- **warmup_ratio (0.1):** first 10% of steps ramp LR from 0 → peak (stabilizes early training).
- **epoch:** one full pass over the train set.
- **batch_size=1 + grad_accum=2:** gradients are summed over 2 micro-batches before each optimizer step → effective batch size = 2.
- **AdamW:** Adam optimizer with decoupled weight decay; "fused" = faster CUDA kernel.
- **weight_decay=0:** no L2-style regularization on weights (LoRA itself is small, so often skipped).
- **max_grad_norm=1.0:** clips gradients so their norm ≤ 1 (prevents explosions).
- **bf16 (bfloat16):** 16-bit float format for faster training with wide dynamic range.
- **gradient_checkpointing:** re-computes activations during backward pass to save VRAM (trades compute for memory).
- **seed=42:** RNG seed for reproducibility.
- **train loss:** loss on training batches (how well the model fits data it sees).
- **eval loss:** loss on held-out eval set (how well it generalizes).
- **grad_norm:** L2 norm of gradients before clipping (proxy for update magnitude).
- **total steps = 120:** = ceil(1120 samples / effective_bs 2) × 3 epochs.

## Loss Curve

| Step | Epoch | Train Loss | Eval Loss | Grad Norm |
|------|-------|------------|-----------|-----------|
| 10   | 0.25  | 7.4358     | —         | 10.39     |
| 20   | 0.50  | 4.3753     | —         | 4.25      |
| 30   | 0.75  | 2.8920     | —         | 3.06      |
| 40   | 1.00  | 1.9366     | **1.9736** | 1.62      |
| 60   | 1.50  | 1.7081     | —         | 1.42      |
| 80   | 2.00  | 1.7641     | **1.8563** | 1.51      |
| 100  | 2.50  | 1.7080     | —         | 1.38      |
| 120  | 3.00  | 1.5893     | **1.8459** | 1.51      |

## Observations
- **Fast convergence in epoch 1**: loss drops 7.44 → 1.94 (~74%).
- **Eval loss improves steadily**: 1.974 → 1.856 → 1.846.
- **No overfitting yet**: train (1.59) vs eval (1.85) gap is modest and eval still decreasing.
- **Gradients stabilized** after step 30 (norm ~1.4–1.6).
- **LR schedule** (cosine) ended near zero — training is fully consumed.

## Conclusion
Training is healthy. Checkpoint-120 is the best; adapter is ready for evaluation. If further gains are needed, consider +1 epoch or a larger train set (eval loss still trending down).
