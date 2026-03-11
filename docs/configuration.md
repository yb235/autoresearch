# Configuration

All configuration in autoresearch is done through **constants in code**. There are no config files, no CLI arguments (for training), no environment variables (beyond PyTorch internals). This is intentional — the agent modifies the code directly.

## Fixed Constants (prepare.py — DO NOT MODIFY)

These define the experiment rules and cannot be changed:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_SEQ_LEN` | 2048 | Context length for training and evaluation |
| `TIME_BUDGET` | 300 | Training time budget in seconds (5 minutes) |
| `EVAL_TOKENS` | 20,971,520 | Validation tokens (~20M = 40 × 524,288) |
| `VOCAB_SIZE` | 8192 | BPE vocabulary size |
| `VAL_SHARD` | 6542 | Pinned validation shard index |
| `MAX_SHARD` | 6542 | Last available data shard |

**Rationale:** These are fixed to ensure every experiment is evaluated on the same terms. Changing `EVAL_TOKENS` would change the evaluation precision. Changing `TIME_BUDGET` would break comparability. Changing `VOCAB_SIZE` would require retraining the tokenizer.

## Tunable Hyperparameters (train.py — Agent Edits These)

### Model Architecture

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `DEPTH` | 8 | Number of transformer layers | More layers = more capacity but slower |
| `ASPECT_RATIO` | 64 | Model dim = depth × aspect_ratio | Controls width vs. depth tradeoff |
| `HEAD_DIM` | 128 | Dimension per attention head | Usually fixed at 64 or 128 |
| `WINDOW_PATTERN` | `"SSSL"` | Attention window pattern | `"L"` = full context, `"S"` = half context |

**How model dimensions are derived:**
```
base_dim  = DEPTH × ASPECT_RATIO = 8 × 64 = 512
model_dim = round_up(base_dim, HEAD_DIM) = 512  (already multiple of 128)
num_heads = model_dim / HEAD_DIM = 512 / 128 = 4
```

**Rationale for ASPECT_RATIO:** Instead of setting width directly, width is derived from depth × aspect_ratio. This means changing `DEPTH` automatically adjusts width proportionally, keeping the model balanced. The agent can tune one knob (`DEPTH`) for model size and another (`ASPECT_RATIO`) for the shape.

### Optimization

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `TOTAL_BATCH_SIZE` | 524,288 (2¹⁹) | Tokens per optimizer step | Larger = more stable but fewer steps in 5 min |
| `DEVICE_BATCH_SIZE` | 128 | Sequences per micro-batch | Limited by VRAM |
| `EMBEDDING_LR` | 0.6 | LR for token embeddings | High because embeddings need fast updates |
| `UNEMBEDDING_LR` | 0.004 | LR for output head | Low because head should change slowly |
| `MATRIX_LR` | 0.04 | LR for weight matrices (Muon) | Main knob for Muon |
| `SCALAR_LR` | 0.5 | LR for per-layer scalars | For resid_lambdas and x0_lambdas |
| `WEIGHT_DECAY` | 0.2 | Muon weight decay | Regularization strength |
| `ADAM_BETAS` | (0.8, 0.95) | Adam momentum coefficients | Lower than standard for fast adaptation |
| `WARMUP_RATIO` | 0.0 | Fraction of time for LR warmup | 0 = no warmup |
| `WARMDOWN_RATIO` | 0.5 | Fraction of time for LR cooldown | Last 50% of training |
| `FINAL_LR_FRAC` | 0.0 | Final LR as fraction of peak | 0 = decay to zero |

### Derived Values

These are computed from the above, not set directly:

| Value | Formula | Default |
|-------|---------|---------|
| `model_dim` | `round_up(DEPTH × ASPECT_RATIO, HEAD_DIM)` | 512 |
| `num_heads` | `model_dim / HEAD_DIM` | 4 |
| `grad_accum_steps` | `TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE × MAX_SEQ_LEN)` | 2 |
| `dmodel_lr_scale` | `(model_dim / 768)^(-0.5)` | ~1.22 |

## Model Internal Constants

These are hardcoded within the model architecture:

| Constant | Value | Where | Description |
|----------|-------|-------|-------------|
| `softcap` | 15 | `GPT.forward()` | Logit soft-capping threshold |
| `ve_gate_channels` | 32 | `CausalSelfAttention` | Input channels for VE gate |
| `MLP expansion` | 4× | `MLP.__init__()` | Hidden dim = 4 × n_embd |
| `RoPE base` | 10,000 | `_precompute_rotary_embeddings()` | Rotary embedding frequency base |
| `rotary_seq_len` | 10 × seq_len | `GPT.__init__()` | Pre-computed RoPE length |

## Optimizer Internal Constants

| Constant | Value | Where | Description |
|----------|-------|-------|-------------|
| `ns_steps` | 5 | Muon param groups | Newton-Schulz iterations for polar decomposition |
| `Muon momentum range` | 0.85 → 0.95 | `get_muon_momentum()` | Ramps over 300 steps |
| `Muon beta2` | 0.95 | Muon param groups | Second moment coefficient for NorMuon |
| `x0_lambdas betas` | (0.96, 0.95) | Setup optimizer | Special betas for x0 scalars |
| `eps` | 1e-10 | All AdamW groups | Adam epsilon |

## Environment Variables

Set automatically at the top of `train.py`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA memory fragmentation |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Silences HuggingFace download bars |

## Hardware Assumptions

| Assumption | Value | Configurable? |
|------------|-------|---------------|
| GPU type | NVIDIA (CUDA) | No (hardcoded) |
| Peak FLOPS | 989.5 TFLOPS (H100 bf16) | Hardcoded for MFU calculation |
| Flash Attention | FA3 (Hopper) or kernels-community (non-Hopper) | Auto-detected |
| Precision | bfloat16 | Hardcoded |

**Rationale for H100 assumption:** The MFU calculation uses H100 peak FLOPS. On other GPUs, MFU will be "wrong" (likely >100% on faster GPUs or very low on slower ones), but this doesn't affect training — it's just a monitoring metric.

## prepare.py CLI Arguments

Only used when running `prepare.py` directly (one-time setup):

```bash
uv run prepare.py [--num-shards N] [--download-workers W]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-shards` | 10 | Training shards to download (-1 = all 6542) |
| `--download-workers` | 8 | Parallel download threads |

## What the Agent Should Tune

Based on the README's recommendations for different hardware:

| If You Want To... | Tune This |
|-------------------|-----------|
| Change model size | `DEPTH` (primary knob) |
| Change model shape | `ASPECT_RATIO` |
| Reduce memory usage | `DEVICE_BATCH_SIZE` (lower), `DEPTH` (lower) |
| Speed up training | `TOTAL_BATCH_SIZE` (lower = more steps), `WINDOW_PATTERN` |
| Improve convergence | Learning rates, `ADAM_BETAS`, warmdown schedule |
| Try different architectures | Modify GPT/Block/MLP classes in `train.py` |
