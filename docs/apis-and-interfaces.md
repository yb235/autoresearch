# APIs & Interfaces

This document describes every public function, class, and interface in the codebase. Autoresearch has no external APIs (no REST endpoints, no server). All interfaces are internal Python module boundaries.

## prepare.py — Public Interface

### Constants

These constants are imported by `train.py` and define the fixed experiment parameters:

```python
MAX_SEQ_LEN = 2048        # Context length for training and evaluation
TIME_BUDGET = 300          # Training time in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # ~20M tokens for validation evaluation
VOCAB_SIZE = 8192          # BPE vocabulary size
```

### `Tokenizer` Class

Wrapper around `tiktoken.Encoding` for tokenization.

```python
class Tokenizer:
    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR) -> Tokenizer
        """Load tokenizer from disk. Default: ~/.cache/autoresearch/tokenizer/"""

    def get_vocab_size(self) -> int
        """Returns vocabulary size (8192)."""

    def get_bos_token_id(self) -> int
        """Returns the BOS token ID."""

    def encode(self, text: str | list[str], prepend: int | str | None = None,
               num_threads: int = 8) -> list[int] | list[list[int]]
        """
        Encode text to token IDs.

        Args:
            text: Single string or list of strings to encode
            prepend: Token ID or token string to prepend to each encoded sequence
            num_threads: Thread count for batch encoding

        Returns:
            List of token IDs (if text is str) or list of lists (if text is list)
        """

    def decode(self, ids: list[int]) -> str
        """Decode token IDs back to text."""
```

**Usage in train.py:**
```python
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
```

### `make_dataloader()` Function

```python
def make_dataloader(tokenizer: Tokenizer, B: int, T: int, split: str,
                    buffer_size: int = 1000) -> Generator
    """
    Create an infinite dataloader with best-fit document packing.

    Args:
        tokenizer: Tokenizer instance
        B: Batch size (number of sequences per batch)
        T: Sequence length (MAX_SEQ_LEN)
        split: "train" or "val"
        buffer_size: Number of documents to buffer for best-fit packing

    Yields:
        tuple of (inputs, targets, epoch):
            inputs:  torch.Tensor[B, T] int64, CUDA — input token IDs
            targets: torch.Tensor[B, T] int64, CUDA — target token IDs
            epoch:   int — current data epoch number
    """
```

**Usage in train.py:**
```python
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
```

### `evaluate_bpb()` Function

```python
@torch.no_grad()
def evaluate_bpb(model: nn.Module, tokenizer: Tokenizer, batch_size: int) -> float
    """
    Compute bits-per-byte on the validation set.

    The model must implement:
        model(x, y, reduction='none') → loss_per_token [B*T]

    Args:
        model: The GPT model (in eval mode)
        tokenizer: Tokenizer instance
        batch_size: Evaluation batch size

    Returns:
        float: Bits per byte (lower is better)
    """
```

**Usage in train.py:**
```python
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

**Model contract:** The model passed to `evaluate_bpb` must support being called as `model(x, y, reduction='none')` and return per-token losses as a flat tensor. This is the interface contract between `prepare.py` and `train.py`.

### `get_token_bytes()` Function

```python
def get_token_bytes(device: str = "cpu") -> torch.Tensor
    """
    Load token byte-length lookup table.

    Returns:
        torch.Tensor[vocab_size] int32 — UTF-8 byte length per token ID
        Special tokens have byte length 0.
    """
```

### Internal Functions (Not Imported by train.py)

These are used only within `prepare.py`:

| Function | Purpose |
|----------|---------|
| `download_single_shard(index)` | Download one parquet shard with retries |
| `download_data(num_shards, download_workers)` | Orchestrate parallel shard downloads |
| `list_parquet_files()` | List all parquet files in data directory |
| `text_iterator(max_chars, doc_cap)` | Yield documents for tokenizer training |
| `train_tokenizer()` | Train and save BPE tokenizer |
| `_document_batches(split, tokenizer_batch_size)` | Infinite document batch iterator |

## train.py — Public Interface

### GPTConfig Dataclass

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048     # Max sequence length
    vocab_size: int = 32768      # Vocabulary size
    n_layer: int = 12            # Number of transformer layers
    n_head: int = 6              # Number of query attention heads
    n_kv_head: int = 6           # Number of key/value heads (for GQA)
    n_embd: int = 768            # Model embedding dimension
    window_pattern: str = "SSSL" # Sliding window attention pattern
```

### GPT Model Class

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig)
        """Build the GPT model from config."""

    def init_weights(self) -> None
        """Initialize all weights with the prescribed strategy."""

    def estimate_flops(self) -> int
        """Estimated FLOPs per token (forward + backward)."""

    def num_scaling_params(self) -> dict[str, int]
        """Parameter count breakdown by category."""

    def setup_optimizer(self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        adam_betas: tuple = (0.8, 0.95),
        scalar_lr: float = 0.5
    ) -> MuonAdamW
        """Create the MuonAdamW optimizer with parameter groups."""

    def forward(self, idx: Tensor, targets: Tensor | None = None,
                reduction: str = 'mean') -> Tensor
        """
        Forward pass.

        Args:
            idx: Input token IDs [B, T]
            targets: Target token IDs [B, T] (None for inference)
            reduction: 'mean' for training, 'none' for evaluation

        Returns:
            If targets provided: loss (scalar if reduction='mean',
                                       [B*T] if reduction='none')
            If no targets: logits [B, T, vocab_size]
        """
```

### MuonAdamW Optimizer Class

```python
class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D params, AdamW for others."""

    def __init__(self, param_groups: list[dict])
    def step(self) -> None
        """Execute one optimization step (dispatches to _step_adamw or _step_muon)."""
```

### Helper Functions

```python
def norm(x: Tensor) -> Tensor
    """RMS normalization on the last dimension."""

def has_ve(layer_idx: int, n_layer: int) -> bool
    """Returns True if layer should have a value embedding (alternating pattern)."""

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor
    """Apply rotary position embeddings to Q or K tensor."""

def build_model_config(depth: int) -> GPTConfig
    """Build a GPTConfig from depth using ASPECT_RATIO and HEAD_DIM."""

def get_lr_multiplier(progress: float) -> float
    """Compute LR multiplier from training progress (0.0 to 1.0)."""

def get_muon_momentum(step: int) -> float
    """Compute Muon momentum for given step (ramps 0.85 → 0.95)."""

def get_weight_decay(progress: float) -> float
    """Compute weight decay from training progress (linear decay to 0)."""
```

## Interface Between Files

The critical interface between `prepare.py` and `train.py`:

```
prepare.py exports:
  ├── MAX_SEQ_LEN (constant)
  ├── TIME_BUDGET (constant)
  ├── Tokenizer (class)
  ├── make_dataloader (function)
  └── evaluate_bpb (function)

train.py imports and uses:
  from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

train.py must provide a model that satisfies:
  model(x, y, reduction='none') → per-token losses [B*T]
  model(x, y)                   → mean loss (scalar)
```

This is the **only** interface contract. As long as `train.py` exports a model that can be called with these signatures, `evaluate_bpb` will work correctly.

## External Dependencies Used

| Package | Import | Used For |
|---------|--------|----------|
| `torch` | Throughout | Neural network framework |
| `torch.nn.functional` | `F.rms_norm`, `F.relu`, `F.cross_entropy` | Operations |
| `kernels` | `get_kernel` | Loading Flash Attention 3 |
| `tiktoken` | Tokenizer encoding | Fast BPE tokenization |
| `rustbpe` | Tokenizer training | Fast BPE vocabulary training |
| `pyarrow.parquet` | `pq.ParquetFile` | Reading parquet data files |
| `requests` | `requests.get` | Downloading data shards |
| `numpy` | (via analysis.ipynb) | Numerical analysis |
| `pandas` | (via analysis.ipynb) | Data analysis |
| `matplotlib` | (via analysis.ipynb) | Plotting results |
