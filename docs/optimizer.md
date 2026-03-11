# Optimizer & Training

This document explains the MuonAdamW optimizer, learning rate schedules, and the training loop in detail.

## The MuonAdamW Optimizer

Autoresearch uses a **hybrid optimizer** that applies different optimization algorithms to different parameter types:

| Parameter Type | Optimizer | Examples |
|---------------|-----------|----------|
| 2D weight matrices | **Muon** | Q, K, V, output projections, MLP weights |
| 1D/scalar/embedding params | **AdamW** | Token embeddings, LM head, value embeddings, per-layer scalars |

**Rationale:** Muon (an optimizer based on orthogonal gradient projection) works exceptionally well for matrix parameters but isn't suitable for embeddings or scalar parameters. AdamW is the reliable workhorse for everything else. Combining them gets the best of both worlds.

## How Parameters Are Grouped

The optimizer creates these parameter groups:

| Group | Kind | Parameters | Base LR |
|-------|------|-----------|---------|
| 1 | AdamW | `lm_head` weights | `UNEMBEDDING_LR` (0.004) × scale |
| 2 | AdamW | Token embedding (`wte`) | `EMBEDDING_LR` (0.6) × scale |
| 3 | AdamW | Value embeddings | `EMBEDDING_LR` (0.6) × scale |
| 4 | AdamW | `resid_lambdas` | `SCALAR_LR × 0.01` (0.005) |
| 5 | AdamW | `x0_lambdas` | `SCALAR_LR` (0.5) |
| 6+ | Muon | Matrix params (grouped by shape) | `MATRIX_LR` (0.04) |

**LR scaling:** All AdamW learning rates are scaled by `(model_dim / 768)^(-0.5)`. This means:
- At 768 dim: scale = 1.0 (the tuned baseline)
- Larger models get smaller LR
- Smaller models get larger LR

**Rationale:** The `1/√d` LR scaling is derived from µP (maximal update parameterization) theory, which predicts that optimal learning rates scale inversely with the square root of the model dimension. This lets hyperparameters transfer across model sizes.

## AdamW Step (Detail)

The AdamW implementation is compiled with `torch.compile` for performance:

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Weight decay (applied before update)
    p.mul_(1 - lr_t * wd_t)

    # Update moving averages
    exp_avg.lerp_(grad, 1 - beta1_t)           # First moment (mean)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)  # Second moment (variance)

    # Bias correction
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t

    # Adam update
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

**Configuration:**
- `beta1 = 0.8` (default is 0.9 in standard Adam — lower means more responsive to recent gradients)
- `beta2 = 0.95` (default is 0.999 — lower means adapting to gradient magnitude faster)
- `eps = 1e-10` (very small — trust the adaptive LR more)
- `weight_decay = 0.0` for all AdamW groups (no weight decay on embeddings/scalars)

**Rationale for lower betas:** In short training runs (5 minutes), the optimizer needs to adapt quickly. Lower beta1 (0.8 vs 0.9) makes the momentum more responsive. Lower beta2 (0.95 vs 0.999) makes the adaptive learning rate adjust faster. These are tuned for the rapid-iteration autoresearch setting.

## Muon Step (Detail)

Muon is a more exotic optimizer. Here's what it does, step by step:

### Step 1: Nesterov Momentum

```python
momentum_buffer.lerp_(stacked_grads, 1 - momentum)
g = stacked_grads.lerp_(momentum_buffer, momentum)
```

Standard Nesterov momentum, where `momentum` ramps from 0.85 to 0.95 over 300 steps.

### Step 2: Polar Express Orthogonalization

```python
X = g.bfloat16()
X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

# For tall matrices (rows >= cols):
for a, b, c in polar_express_coeffs[:ns_steps]:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
```

This is the core of Muon. It projects the gradient onto the nearest orthogonal matrix using **polar decomposition** via Newton-Schulz iterations.

**What this means intuitively:** Instead of stepping in the direction of the gradient, Muon steps in the direction of the "orthogonalized gradient." This has a normalizing effect — it treats all directions in weight space more equally, preventing some directions from dominating training.

**Coefficients:**
```python
polar_express_coeffs = [
    (8.157, -22.483, 15.879),
    (4.043, -2.809, 0.500),
    (3.892, -2.772, 0.506),
    (3.286, -2.368, 0.464),
    (2.347, -1.710, 0.423),
]
```

These are pre-computed polynomial coefficients for a fast Newton-Schulz iteration. Using 5 steps (`ns_steps=5`) gives a very good approximation to the true orthogonal projection.

**Tall vs. wide matrices:** The algorithm branches based on matrix shape:
- Tall matrices (`rows >= cols`): compute `X.mT @ X` (cheaper when cols < rows)
- Wide matrices (`rows < cols`): compute `X @ X.mT` (cheaper when rows < cols)

### Step 3: NorMuon Variance Reduction

```python
v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
second_momentum_buffer.lerp_(v_mean, 1 - beta2)
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
```

This adds per-dimension adaptive scaling (similar to Adam's second moment) on top of the orthogonal gradient. It normalizes the update across the non-reduced dimension.

**Rationale:** Pure Muon treats all matrix rows/columns equally. NorMuon adds adaptive scaling so rows that consistently have larger gradients get smaller updates, similar to how Adam adapts per-parameter.

### Step 4: Cautious Weight Decay + Update

```python
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

- **Cautious weight decay**: only applies weight decay to parameters where the gradient and parameter have the same sign
- This prevents weight decay from fighting the gradient

**Rationale:** Standard weight decay blindly shrinks all parameters. Cautious weight decay only shrinks parameters that the gradient would naturally shrink anyway, avoiding the counterproductive case where decay pushes a parameter away from where the gradient wants it to go.

### Muon LR Scaling

```python
group["lr"] = group["initial_lr"] * max(1.0, shape[-2] / shape[-1])**0.5
```

For tall matrices, the learning rate is scaled up by `√(rows/cols)`. Wide matrices keep the base LR.

**Rationale:** After orthogonalization, tall matrices naturally have smaller updates per row (the orthogonality constraint is stronger). Scaling by `√(aspect ratio)` compensates for this.

## Learning Rate Schedules

All schedules are based on `progress = training_time / TIME_BUDGET`:

### Main LR Schedule

```
LR multiplier
    1.0 ─────────────────────────┐
                                 │
                                  \
                                   \
                                    \
    0.0 ──────────────────────────────┐
    |   warmup   |   constant   | warmdown |
    0%         0%             50%        100%
```

- **Warmup** (`WARMUP_RATIO = 0.0`): None by default (immediate full LR)
- **Constant phase**: First 50% of training at full LR
- **Warmdown** (`WARMDOWN_RATIO = 0.5`): Linear decay to `FINAL_LR_FRAC = 0.0` over the last 50%

**Rationale:** No warmup because `torch.compile` startup steps are excluded from the time budget. The long warmdown (50% of training) implements a cosine-decay-like schedule that helps the model converge at the end. Setting final LR to 0 ensures the model fully settles.

### Muon Momentum Schedule

```python
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
```

- Starts at 0.85, linearly increases to 0.95 over the first 300 steps
- Stays at 0.95 afterwards

**Rationale:** Lower initial momentum helps the optimizer explore early. As training progresses and the loss landscape smooths out, higher momentum provides better convergence.

### Weight Decay Schedule

```python
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
```

- Starts at `WEIGHT_DECAY = 0.2`, linearly decreases to 0
- Only applied to Muon parameters (all AdamW groups have `weight_decay = 0.0`)

**Rationale:** Decreasing weight decay over training makes intuitive sense — early on, you want regularization to prevent overfitting to initial data. Later, you want the model to fully utilize its capacity. This is sometimes called "weight decay warmdown."

## The Training Loop

```python
while True:
    # 1. Forward + backward (with gradient accumulation)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:          # bfloat16 mixed precision
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)  # prefetch next batch

    # 2. Compute schedules from wall clock progress
    progress = training_time / TIME_BUDGET
    lr_multiplier = get_lr_multiplier(progress)
    momentum = get_muon_momentum(step)
    weight_decay = get_weight_decay(progress)

    # 3. Apply schedule to all parameter groups
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_multiplier
        ...

    # 4. Optimizer step
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # 5. Fast-fail check
    if train_loss > 100:
        print("FAIL")
        exit(1)

    # 6. Stop when time budget exhausted (after warmup steps)
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
```

### Gradient Accumulation

```
TOTAL_BATCH_SIZE = 2^19 = 524,288 tokens
DEVICE_BATCH_SIZE = 128
MAX_SEQ_LEN = 2048

tokens_per_fwdbwd = 128 × 2048 = 262,144
grad_accum_steps = 524,288 / 262,144 = 2
```

So each optimizer step processes 2 micro-batches before updating weights.

**Rationale:** Large batch sizes help with training stability and convergence, but they require more memory. Gradient accumulation achieves the effective large batch size (524K tokens) while only needing memory for the device batch size (128 sequences × 2048 tokens).

### Mixed Precision (bfloat16)

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
```

- Forward pass runs in bfloat16
- Gradients are computed in bfloat16
- Optimizer states are in the parameter's native dtype
- No loss scaling needed (bfloat16 has sufficient dynamic range)

**Rationale:** bfloat16 halves memory usage and doubles throughput on modern GPUs (H100 has dedicated bf16 tensor cores). Unlike float16, bfloat16 has the same exponent range as float32, so it doesn't need loss scaling for stable training.

### GC Management

```python
if step == 0:
    gc.collect()
    gc.freeze()
    gc.disable()
elif (step + 1) % 5000 == 0:
    gc.collect()
```

- After the first step: collect garbage, freeze current objects, disable GC
- Every 5000 steps: manual collection

**Rationale:** Python's garbage collector causes unpredictable ~500ms stalls during training. By disabling it and only running it rarely, training steps have consistent timing. `gc.freeze()` tells the GC to not scan objects that existed at freeze time, further reducing overhead.

### Compilation Warmup

```python
if step > 10 and total_training_time >= TIME_BUDGET:
    break
```

The first 10 steps don't count toward `total_training_time`. This is because `torch.compile` needs to trace and compile the model on the first few forward passes, which is much slower than normal execution.

**Rationale:** Excluding compilation time ensures the 5-minute budget measures actual training work. Without this, the first run on a fresh compilation would be unfairly penalized.

## MFU (Model FLOP Utilization)

```python
H100_BF16_PEAK_FLOPS = 989.5e12  # ~990 teraFLOP/s

steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
                   / total_training_time / H100_BF16_PEAK_FLOPS
```

MFU tells you what percentage of the GPU's theoretical peak performance you're actually using.

- **40% MFU** is typical for this type of workload
- Higher MFU means more efficient use of hardware
- MFU < 30% suggests a bottleneck (data loading, memory bandwidth, etc.)

**Rationale:** MFU is the standard efficiency metric for LLM training. It helps identify whether changes improve or harm hardware utilization.
