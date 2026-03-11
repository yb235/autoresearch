# Data Schema

This document describes every data format, file structure, and in-memory data representation used in autoresearch.

## On-Disk Data

### Parquet Shards

**Location:** `~/.cache/autoresearch/data/`

**Naming:** `shard_NNNNN.parquet` (zero-padded 5-digit index, e.g., `shard_00000.parquet`)

**Source:** `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle`

**Schema (per parquet file):**

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Raw document text |

Each parquet file contains multiple row groups. Each row group contains a batch of documents in the `text` column. The files are part of a shuffled 400-billion-token web text mix.

**Special shards:**
- Shards 0 to 6541: Training data
- Shard 6542 (`shard_06542.parquet`): Pinned validation shard — always used for evaluation, never for training

**Rationale:** Parquet is a columnar format that's fast to read and compressed. Using HuggingFace's dataset hosting provides reliable CDN-backed downloads.

### Tokenizer Files

**Location:** `~/.cache/autoresearch/tokenizer/`

| File | Format | Description |
|------|--------|-------------|
| `tokenizer.pkl` | Python pickle | A `tiktoken.Encoding` object containing the trained BPE tokenizer |
| `token_bytes.pt` | PyTorch tensor | `int32` tensor of shape `[vocab_size]` — UTF-8 byte length per token |

**Tokenizer details:**

| Property | Value |
|----------|-------|
| Vocabulary size | 8,192 |
| Merge tokens | 8,188 |
| Special tokens | 4 (`<\|reserved_0\|>` through `<\|reserved_3\|>`) |
| BOS token | `<\|reserved_0\|>` |
| Split pattern | GPT-4 style regex (see below) |

**Split pattern** (determines how text is pre-split before BPE):
```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

This handles English contractions, Unicode letters, numbers (1–2 digits), punctuation, and whitespace. It's nearly identical to GPT-4's pattern but caps numbers at 2 digits instead of 3.

**token_bytes.pt structure:**
```
Index 0:     byte_length of token 0
Index 1:     byte_length of token 1
...
Index 8191:  byte_length of token 8191

Special tokens → byte_length = 0 (excluded from BPB calculation)
Regular tokens → byte_length = len(decoded_text.encode("utf-8"))
```

**Rationale:** The `token_bytes.pt` lookup table enables efficient BPB evaluation. Instead of decoding each token to count its bytes during eval, we pre-compute and look them up in O(1).

### Results TSV

**Location:** `results.tsv` (project root, git-untracked)

**Format:** Tab-separated values (NOT comma-separated — commas can appear in descriptions)

**Schema:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `commit` | string | `a1b2c3d` | Short (7-char) git commit hash |
| `val_bpb` | float | `0.997900` | Validation bits per byte (0.0 for crashes) |
| `memory_gb` | float | `44.0` | Peak VRAM in GB (0.0 for crashes) |
| `status` | string | `keep` | One of: `keep`, `discard`, `crash` |
| `description` | string | `increase LR to 0.04` | Short text describing the experiment |

**Example:**
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

**Rationale:** TSV over CSV because experiment descriptions might contain commas. The file is untracked by git so the agent can freely append to it without creating merge conflicts.

## In-Memory Data Structures

### GPTConfig

A Python `dataclass` that defines the model shape:

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048    # Maximum context length
    vocab_size: int = 32768     # Size of token vocabulary
    n_layer: int = 12           # Number of transformer blocks
    n_head: int = 6             # Number of attention heads (query)
    n_kv_head: int = 6          # Number of key/value heads (for GQA)
    n_embd: int = 768           # Model dimension (embedding size)
    window_pattern: str = "SSSL" # Attention window pattern
```

In practice, these defaults are overridden by `build_model_config()` which derives them from the `DEPTH` and `ASPECT_RATIO` hyperparameters:

```
model_dim = DEPTH × ASPECT_RATIO  (rounded up to multiple of HEAD_DIM)
num_heads = model_dim / HEAD_DIM
```

### Dataloader Tensors

The dataloader yields a tuple of `(inputs, targets, epoch)`:

| Tensor | Shape | Dtype | Device | Description |
|--------|-------|-------|--------|-------------|
| `inputs` | `[B, T]` | `int64` | CUDA | Input token IDs |
| `targets` | `[B, T]` | `int64` | CUDA | Target token IDs (shifted by 1) |
| `epoch` | `int` | — | CPU | Current data epoch |

Where:
- `B` = `DEVICE_BATCH_SIZE` (default 128)
- `T` = `MAX_SEQ_LEN` (2048)

The relationship between inputs and targets:
```
Row buffer:  [tok0, tok1, tok2, ..., tok2048]  (T+1 = 2049 tokens)
inputs:      [tok0, tok1, tok2, ..., tok2047]  (first T tokens)
targets:     [tok1, tok2, tok3, ..., tok2048]  (last T tokens)
```

### Internal Memory Layout

The dataloader uses a triple-buffer scheme for efficient CPU→GPU transfer:

```
row_buffer:   [B, T+1] int64  — CPU, packing workspace
cpu_buffer:   [2*B*T]  int64  — CPU, pinned memory
gpu_buffer:   [2*B*T]  int64  — CUDA

cpu_inputs  = cpu_buffer[:B*T].view(B, T)
cpu_targets = cpu_buffer[B*T:].view(B, T)
gpu_inputs  = gpu_buffer[:B*T].view(B, T)
gpu_targets = gpu_buffer[B*T:].view(B, T)
```

**Rationale:** Pinned memory enables asynchronous CPU→GPU transfers. The flat buffer with views avoids memory fragmentation.

### Training Output Summary

When training finishes, it prints these metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `val_bpb` | float | Validation bits per byte (THE metric — lower is better) |
| `training_seconds` | float | Wall clock time spent on training steps |
| `total_seconds` | float | Total runtime including startup and evaluation |
| `peak_vram_mb` | float | Peak GPU memory allocated in MB |
| `mfu_percent` | float | Model FLOP Utilization (% of theoretical H100 peak) |
| `total_tokens_M` | float | Total tokens processed in millions |
| `num_steps` | int | Number of optimizer steps completed |
| `num_params_M` | float | Total model parameters in millions |
| `depth` | int | Number of transformer layers |

**Rationale:** This fixed output format makes it easy for the agent to parse results with `grep "^val_bpb:" run.log` and programmatically decide whether to keep or discard an experiment.
