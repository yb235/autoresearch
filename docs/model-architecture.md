# Model Architecture

The autoresearch model is a GPT (decoder-only transformer) with several modern techniques. This document explains every component from the ground up.

## Overview

```
Token IDs [B, T]
      │
      ▼
┌──────────────────────────┐
│   Token Embedding (wte)  │  nn.Embedding(vocab_size, n_embd)
│   + RMS Normalization    │
└───────────┬──────────────┘
            │
     x0 = x (saved)
            │
     ┌──────┴──────┐
     │  Block 0    │──► λ_resid[0]*x + λ_x0[0]*x0 → Attention → MLP
     │  Block 1    │──► λ_resid[1]*x + λ_x0[1]*x0 → Attention → MLP
     │  ...        │
     │  Block N-1  │──► λ_resid[N-1]*x + λ_x0[N-1]*x0 → Attention → MLP
     └──────┬──────┘
            │
     ┌──────┴──────────────┐
     │  RMS Normalization  │
     │  Linear Head        │  nn.Linear(n_embd, vocab_size)
     │  Logit Soft-Cap     │  15 × tanh(logits / 15)
     └───────────┬─────────┘
                 │
                 ▼
        Logits [B, T, vocab_size]
```

## Default Configuration

With the default `DEPTH=8` and `ASPECT_RATIO=64`:

| Parameter | Value | How It's Derived |
|-----------|-------|------------------|
| `n_layer` | 8 | `DEPTH` |
| `n_embd` | 512 | `DEPTH × ASPECT_RATIO = 512`, rounded up to multiple of `HEAD_DIM` |
| `n_head` | 4 | `n_embd / HEAD_DIM = 512/128` |
| `n_kv_head` | 4 | Same as `n_head` (no GQA by default) |
| `head_dim` | 128 | `HEAD_DIM` constant |
| `sequence_len` | 2048 | From `prepare.py: MAX_SEQ_LEN` |
| `vocab_size` | 8,192 | From tokenizer |
| `window_pattern` | "SSSL" | Short-Short-Short-Long attention pattern |
| ~Total params | ~50M | Varies by config |

## Component Deep Dive

### 1. Token Embedding + Normalization

```python
x = self.transformer.wte(idx)   # [B, T] → [B, T, n_embd]
x = norm(x)                     # RMS normalization
```

- Standard learned embedding table
- Immediately RMS-normalized after lookup
- Initialized with `N(0, 1)` (standard normal)
- Cast to `bfloat16` after initialization for memory efficiency

**Rationale:** Normalizing embeddings early stabilizes training. The standard normal initialization (rather than the small init often used) works well with the subsequent normalization.

### 2. Differential Residual Connections

Before each transformer block:

```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

- `resid_lambdas`: initialized to 1.0 — the residual stream contribution
- `x0_lambdas`: initialized to 0.1 — the original embedding contribution
- Both are **learnable per-layer scalars**

This is different from standard residual connections (`x = x + block(x)`). Instead, each layer dynamically mixes:
- The current residual stream (what's been computed so far)
- The original embedding (a "shortcut" back to the input)

**Rationale:** This technique (related to ideas from "Differential Transformer" research) helps with gradient flow in deep networks. The x0 shortcut provides a direct path from input to any layer, preventing representation degradation. The learnable scalars let the model decide how much of each to use.

### 3. Attention (CausalSelfAttention)

Each attention layer computes:

```
Input x [B, T, C]
    │
    ├──► c_q: Linear → Q [B, T, n_head, head_dim]
    ├──► c_k: Linear → K [B, T, n_kv_head, head_dim]
    └──► c_v: Linear → V [B, T, n_kv_head, head_dim]
              │
              │  (+ Value Embedding on alternating layers)
              │
    Apply RoPE to Q, K
    RMS normalize Q, K (QK-norm)
              │
    Flash Attention 3 (causal, with window size)
              │
    c_proj: Linear → output [B, T, C]
```

#### 3a. Query/Key/Value Projections

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

- Separate linear projections (no bias) for Q, K, V
- Supports **Grouped Query Attention (GQA)**: `n_kv_head` can be less than `n_head`
- Default config uses `n_kv_head == n_head` (standard multi-head attention)

**Rationale:** GQA reduces KV cache size and computation while maintaining quality. Keeping the option open lets the agent experiment with different head ratios.

#### 3b. Value Embeddings (ResFormer)

On alternating layers, an additional value embedding is mixed into V:

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

**How it works:**
- A separate embedding table maps token IDs → value vectors
- A small gate network (32 input channels → n_kv_head outputs) computes per-head gates
- The gate is `2 × sigmoid(...)`, so it ranges from 0 to 2
- Initialized so `sigmoid(0) = 0.5`, scaled by 2 → **neutral gate = 1.0**
- The value embedding is added to V weighted by this gate

**Which layers have value embeddings?**
```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```
This ensures the **last layer always has a value embedding**, and they alternate from there.

For `n_layer=8`: layers 1, 3, 5, 7 have value embeddings.

**Rationale:** Value embeddings (from the ResFormer paper) provide a residual path for value computations. Instead of computing V solely from the current hidden state, the model can also directly look up a learned value embedding for each input token. The gating mechanism lets the model learn when to use these embeddings. Alternating layers reduces the memory cost while still providing the benefit.

#### 3c. Rotary Position Embeddings (RoPE)

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

RoPE encodes position information by rotating pairs of dimensions in Q and K:

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

- Pre-computed for up to `10 × sequence_len` positions
- Uses standard base frequency of 10,000
- Applied to both Q and K (not V)

**Rationale:** RoPE provides relative position encoding without learned parameters. The rotation makes attention naturally distance-aware: tokens closer together have more similar Q·K products. Pre-computing for 10× the sequence length allows for future extension without re-initialization.

#### 3d. QK Normalization

```python
q, k = norm(q), norm(k)
```

Both Q and K are RMS-normalized after RoPE application.

**Rationale:** QK-norm (from the Gemma/Gemini lineage) prevents attention logits from growing too large, which stabilizes training especially at larger scales. Without this, the dot product `Q·K` can become very large, leading to sharp attention distributions and gradient instability.

#### 3e. Sliding Window Attention

```python
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

The `window_pattern` controls attention span per layer:

| Pattern Char | Window Size | Meaning |
|-------------|-------------|---------|
| `"L"` (Long) | `sequence_len` (2048) | Full context attention |
| `"S"` (Short) | `sequence_len / 2` (1024) | Half context attention |

Default pattern `"SSSL"` repeats across layers:
```
Layer 0: S (1024)
Layer 1: S (1024)
Layer 2: S (1024)
Layer 3: L (2048)
Layer 4: S (1024)
Layer 5: S (1024)
Layer 6: S (1024)
Layer 7: L (2048)  ← last layer always forced to L
```

**Rationale:** Sliding window attention reduces compute for most layers (attention is O(T × window) instead of O(T²)). The pattern ensures periodic full-context layers so the model can still attend to distant tokens. The last layer is always full context so the final representation has access to everything.

### 4. MLP (Feed-Forward)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()    # ReluSquared activation
        x = self.c_proj(x)
        return x
```

- Standard two-layer MLP with 4× expansion
- **ReluSquared activation** (`relu(x)²`) instead of GELU or SwiGLU
- No bias terms anywhere

**Rationale:** ReluSquared (from the "Primer" paper) creates sparser activations than GELU, which can be more compute-efficient and has shown competitive performance. The squaring amplifies already-positive values while keeping the sparsity benefit of ReLU. No bias terms simplify the model and reduce parameter count slightly.

### 5. Block (Putting Attention + MLP Together)

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
```

- Pre-norm architecture: normalize **before** attention/MLP, not after
- Residual connections around both attention and MLP

**Rationale:** Pre-norm (as opposed to post-norm in the original Transformer) makes training more stable, especially for deeper models. The gradients flow more smoothly through the residual connections.

### 6. Output Head

```python
logits = self.lm_head(x)       # Linear projection to vocab size
logits = logits.float()         # Cast to float32 for stability
logits = softcap * torch.tanh(logits / softcap)  # Soft-cap at ±15
```

**Logit soft-capping:**
- Logits are scaled through `15 × tanh(logits/15)`
- This smoothly limits logits to the range (-15, +15)
- Near zero, tanh(x) ≈ x, so small logits are unchanged
- Large logits are compressed toward ±15

**Rationale:** Soft-capping (from Gemma 2) prevents logit explosion, which can cause training instability. The softcap value of 15 is large enough that it doesn't significantly affect normal-range logits but prevents extreme values. Casting to float32 first ensures numerical precision in the softmax that follows.

### 7. Weight Initialization

The model uses carefully chosen initialization:

| Parameters | Init Strategy | Values |
|------------|--------------|--------|
| Token embedding (`wte`) | Normal | `N(0, 1)` |
| LM head (`lm_head`) | Normal | `N(0, 0.001)` — very small |
| Q, K, V projections | Uniform | `U(-s, s)` where `s = √3 / √n_embd` |
| Output projections (`c_proj`) | Zeros | All zeros |
| MLP up-projection (`c_fc`) | Uniform | `U(-s, s)` where `s = √3 / √n_embd` |
| MLP down-projection (`c_proj`) | Zeros | All zeros |
| Residual lambdas | Constant | 1.0 |
| x0 lambdas | Constant | 0.1 |
| Value embeddings | Uniform | `U(-s, s)` where `s = √3 / √n_embd` |
| VE gate weights | Zeros | (sigmoid(0)=0.5, ×2 = 1.0 = neutral) |

**Rationale for zero-init output projections:** This makes each transformer block initially an identity function (output = input). The model starts as a simple lookup table (embedding → head) and gradually learns to use the transformer layers. This is key for stable training from random init.

**Rationale for small lm_head init:** The very small (0.001 std) initialization of the output head means initial logits are near-uniform across the vocabulary, giving a reasonable starting loss without any token being strongly favored.

## Parameter Count Breakdown

The model tracks parameters in categories:

| Category | What's Included |
|----------|----------------|
| `wte` | Token embedding table |
| `value_embeds` | Value embedding tables (alternating layers) |
| `lm_head` | Output projection matrix |
| `transformer_matrices` | All Q, K, V, output proj, MLP weights |
| `scalars` | `resid_lambdas` + `x0_lambdas` |
| `total` | Everything |

The FLOP estimation excludes embeddings and scalars (they're memory-bound, not compute-bound) and accounts for sliding window attention reducing the effective sequence length.
