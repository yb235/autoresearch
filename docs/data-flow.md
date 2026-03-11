# Data Flow

This document traces data through the entire system, from raw text on the internet to the final `val_bpb` metric.

## Phase 1: Data Preparation (One-Time)

Run once with `uv run prepare.py`. This downloads and preprocesses everything.

### Step 1: Download Data Shards

```
HuggingFace (climbmix-400b-shuffle)
         │
         │  HTTP GET (parallel, 8 workers)
         ▼
~/.cache/autoresearch/data/
  ├── shard_00000.parquet
  ├── shard_00001.parquet
  ├── ...
  └── shard_06542.parquet  ← pinned validation shard
```

**What happens:**
- The dataset has 6,543 parquet shards (indices 0–6542)
- By default, 10 training shards are downloaded (configurable via `--num-shards`)
- Shard 6542 is **always** downloaded — it's the pinned validation shard
- Downloads happen in parallel (8 workers) with retry logic (5 attempts, exponential backoff)
- Each shard is downloaded to a `.tmp` file first, then atomically renamed (prevents corrupt partial downloads)

**Rationale:** Pinning a specific validation shard ensures evaluation is always on the same data, making experiments comparable. Using the last shard avoids overlap with training data regardless of how many training shards you download.

### Step 2: Train Tokenizer

```
Training shards (all except shard_06542)
         │
         │  Read text column from parquet
         ▼
   text_iterator()
   (yields documents, up to 1B chars, 10K cap per doc)
         │
         │  rustbpe BPE training
         ▼
~/.cache/autoresearch/tokenizer/
  ├── tokenizer.pkl      (tiktoken Encoding object)
  └── token_bytes.pt     (tensor: bytes per token ID)
```

**What happens:**
1. `text_iterator()` streams documents from training parquet files
   - Caps each document at 10,000 characters
   - Stops after 1 billion total characters
2. `rustbpe` trains a BPE tokenizer (Rust implementation — fast)
   - Vocabulary: 8,192 tokens (8,188 merge tokens + 4 special tokens)
   - Split pattern: GPT-4 style regex
3. The trained tokenizer is wrapped as a `tiktoken.Encoding` and pickled
4. A `token_bytes.pt` tensor is created: maps each token ID to its UTF-8 byte length
   - This is needed for the bits-per-byte (BPB) evaluation metric
   - Special tokens get byte length 0 (excluded from BPB calculation)

**Rationale:** Using `rustbpe` for training (fast, Rust) + `tiktoken` for inference (fast, battle-tested) combines the best of both. The `token_bytes.pt` lookup table allows efficient BPB calculation without decoding tokens during evaluation.

## Phase 2: Training Runtime

When `uv run train.py` runs, data flows through these stages:

### Step 3: Tokenization & Dataloader

```
Parquet files
      │
      │  _document_batches() — infinite iterator
      ▼
Raw text batches (128 docs at a time)
      │
      │  tokenizer.encode() — BPE encoding + BOS prepend
      ▼
Token ID lists (variable length)
      │
      │  make_dataloader() — best-fit packing
      ▼
Fixed-size tensors: inputs[B, T] and targets[B, T]
      │
      │  async GPU transfer (pin_memory → non_blocking copy)
      ▼
GPU tensors ready for training
```

**What happens in detail:**

1. **Document streaming** (`_document_batches()`):
   - Iterates through parquet files, reads the `text` column
   - For training: uses all shards except the validation shard
   - For validation: uses only the validation shard
   - Yields batches of 128 documents at a time
   - Loops infinitely (wraps around with epoch counter)

2. **Tokenization**:
   - Each document batch is encoded using the BPE tokenizer
   - A BOS (beginning of sequence) token is prepended to each document
   - Uses multi-threaded encoding (`num_threads=8`)

3. **Best-fit packing** (`make_dataloader()`):
   - The key challenge: documents have variable lengths, but the model needs fixed-size rows
   - Each row has capacity `T + 1` tokens (2049 for default `MAX_SEQ_LEN=2048`)
   - **Packing algorithm:**
     - Maintain a buffer of ~1000 tokenized documents
     - For each row position: find the **largest** document that fits in the remaining space
     - If no document fits: take the **shortest** document and crop it to fill exactly
   - Result: 100% utilization (no padding tokens), multiple documents per row

4. **Tensor construction**:
   - Row buffer → split into `inputs[:, :-1]` and `targets[:, 1:]` (standard next-token prediction)
   - Copy to pinned CPU memory → async copy to GPU

**Rationale for best-fit packing:** Naive approaches (truncate all docs to T, or simple concatenation with padding) waste tokens. Best-fit packing maximizes the information per training step. The "crop shortest when nothing fits" heuristic ensures zero padding while keeping most documents intact.

### Step 4: Forward Pass

```
inputs[B, T]  (token IDs)
      │
      ▼
┌─ Token Embedding (wte) ──────────────────────────┐
│  nn.Embedding(vocab_size, n_embd) → [B, T, C]    │
│  then RMS normalize                               │
└──────────────────────────┬────────────────────────┘
                           │
      x0 = x  (save for residual shortcut)
                           │
      ┌────────────────────┴────────────────────────┐
      │  For each transformer block i = 0..n_layer: │
      │                                              │
      │  x = λ_resid[i] * x + λ_x0[i] * x0         │
      │       (differential residual mixing)         │
      │                                              │
      │  ┌─ Attention ──────────────────────────┐    │
      │  │ Q, K, V projections                  │    │
      │  │ Value Embedding (on alternating layers)│   │
      │  │ RoPE (rotary position embeddings)    │    │
      │  │ QK normalization                     │    │
      │  │ Flash Attention 3 (with window size) │    │
      │  │ Output projection                    │    │
      │  └──────────────────────────────────────┘    │
      │       x = x + attn(norm(x))                  │
      │                                              │
      │  ┌─ MLP ────────────────────────────────┐    │
      │  │ Linear → ReluSquared → Linear        │    │
      │  └──────────────────────────────────────┘    │
      │       x = x + mlp(norm(x))                   │
      └────────────────────┬────────────────────────┘
                           │
      ┌────────────────────┴────────────────────────┐
      │  Final RMS norm                              │
      │  Linear head → logits[B, T, vocab_size]      │
      │  Soft-cap at ±15 (tanh scaling)              │
      └────────────────────┬────────────────────────┘
                           │
                           ▼
      Cross-entropy loss against targets[B, T]
```

### Step 5: Backward Pass & Optimization

```
loss.backward()
      │
      │  (repeat for grad_accum_steps micro-batches)
      ▼
Accumulated gradients
      │
      ├──► AdamW step (for embeddings, lm_head, scalars)
      │    • Standard Adam with bias correction
      │    • Different LR per parameter group
      │
      └──► Muon step (for 2D weight matrices)
           • Nesterov momentum
           • Polar Express orthogonalization
           • NorMuon variance reduction
           • Cautious weight decay
      │
      ▼
Updated parameters → next step
```

### Step 6: Evaluation

After training completes (5-minute budget exhausted):

```
Validation data (shard_06542 only)
      │
      │  Same dataloader pipeline
      ▼
For EVAL_TOKENS / (B * T) steps:
      │
      │  Forward pass (no gradients)
      ▼
Per-token cross-entropy loss (nats)
      │
      │  × mask (exclude special tokens where byte_length = 0)
      ▼
Sum of nats / (ln(2) × sum of bytes) = val_bpb
```

**Bits per Byte (BPB) calculation:**
- For each token: compute cross-entropy loss (in nats) and look up its byte length
- Sum all nats, sum all bytes (excluding special tokens)
- BPB = total_nats / (ln(2) × total_bytes)
- Lower is better

**Rationale:** BPB is vocab-size-independent. If the agent changes the tokenizer vocabulary, the metric stays comparable. It measures "how many bits does the model need per byte of text?" — a fundamental measure of compression quality.

## Complete Data Flow Summary

```
Internet → Parquet shards → BPE tokenizer training
                │                     │
                ▼                     ▼
         Raw text docs         tokenizer.pkl
                │              token_bytes.pt
                ▼                     │
    Tokenized doc lists ◄─────────────┘
                │
    Best-fit packed rows [B, T+1]
                │
    inputs[B, T] + targets[B, T]
                │
    GPU transfer (pinned memory)
                │
    Forward pass → loss
                │
    Backward pass → gradients
                │
    Optimizer step → updated weights
                │
    (repeat for 5 minutes)
                │
    Final eval → val_bpb (the one number that matters)
```
