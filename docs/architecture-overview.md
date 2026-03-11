# Architecture Overview

## What Is Autoresearch?

Autoresearch is a system that turns an AI coding agent (like Claude or Codex) into an autonomous ML researcher. Instead of a human tweaking hyperparameters and model architectures, the AI agent does it — running experiments back-to-back, keeping improvements, discarding failures, and never stopping until told to.

Think of it as a tireless research intern who runs ~12 experiments per hour while you sleep.

## The Big Picture

```
┌──────────────────────────────────────────────────────────────┐
│                        HUMAN LAYER                           │
│                                                              │
│  The human writes program.md — the "research instructions"   │
│  Then launches an AI agent and goes to sleep                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      AI AGENT LAYER                          │
│                                                              │
│  Reads program.md, then enters an infinite loop:             │
│  1. Come up with an experiment idea                          │
│  2. Edit train.py with the change                            │
│  3. git commit                                               │
│  4. Run: uv run train.py > run.log 2>&1                     │
│  5. Check results (val_bpb)                                  │
│  6. Keep (if improved) or revert (if not)                    │
│  7. Log to results.tsv                                       │
│  8. Repeat forever                                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING LAYER                             │
│                                                              │
│  train.py — single-file, single-GPU training:                │
│  • Builds a GPT model (configurable depth/width/heads)       │
│  • Uses MuonAdamW optimizer (Muon + AdamW hybrid)            │
│  • Trains for exactly 5 minutes (wall clock)                 │
│  • Evaluates on validation set → reports val_bpb             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│                                                              │
│  prepare.py — one-time setup, then runtime utilities:        │
│  • Downloads parquet shards from HuggingFace                 │
│  • Trains a BPE tokenizer (8192 vocab)                       │
│  • Provides dataloader + evaluation function at runtime      │
│  • Stored in ~/.cache/autoresearch/                          │
└──────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### `prepare.py` — The Foundation (Read-Only)

This file does two jobs:

**Job 1: One-time setup** (run once via `uv run prepare.py`)
- Downloads training data shards (parquet files from HuggingFace)
- Trains a BPE tokenizer using `rustbpe`
- Saves everything to `~/.cache/autoresearch/`

**Job 2: Runtime library** (imported by `train.py`)
- `Tokenizer` class — wraps the trained tokenizer
- `make_dataloader()` — infinite data iterator with smart document packing
- `evaluate_bpb()` — the sacred evaluation function (bits per byte)
- Constants — `MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, `VOCAB_SIZE`

**Rationale:** By keeping data prep and evaluation in a separate read-only file, the AI agent can freely hack on `train.py` without accidentally breaking the evaluation metric or data pipeline. This is a critical safety boundary.

### `train.py` — The Experiment (Agent-Modified)

This is the **only file the AI agent edits**. It contains:

- **GPT model** — Transformer with modern techniques (RoPE, RMS norm, GQA, value embeddings, sliding window attention, logit soft-capping)
- **MuonAdamW optimizer** — A hybrid optimizer using Muon for 2D weight matrices and AdamW for everything else
- **Hyperparameters** — All tunable knobs in one place (depth, learning rates, batch sizes, etc.)
- **Training loop** — Time-budgeted training with warmup/warmdown schedules

**Rationale:** Everything in one file means the agent has a single target to modify. No complex multi-file refactoring, no import issues, no config file parsing. Just edit the code and run it.

### `program.md` — The Research Protocol (Human-Modified)

This is the instruction set for the AI agent. It defines:

- How to set up an experiment branch
- The rules of engagement (what can/can't be modified)
- The experiment loop protocol
- How to log results
- The "never stop" directive

**Rationale:** By putting agent instructions in a Markdown file rather than code, the human can iterate on the "research org design" in natural language. The agent reads this like a briefing document and follows the protocol.

## File Dependency Graph

```
program.md ──────────► AI Agent (reads instructions)
                          │
                          │ edits
                          ▼
prepare.py ◄──────── train.py
  (imports)           (the experiment)
     │
     ▼
~/.cache/autoresearch/
  ├── data/            (parquet shards)
  └── tokenizer/       (tokenizer.pkl + token_bytes.pt)
```

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.10+ | ML ecosystem standard |
| Deep Learning | PyTorch 2.9.1 | Industry standard, `torch.compile` support |
| Attention | Flash Attention 3 | Fast fused attention kernel (Hopper GPU optimized) |
| Tokenizer | rustbpe + tiktoken | Fast BPE training (Rust) + fast inference (tiktoken) |
| Data Format | Parquet (PyArrow) | Columnar, compressed, fast reads |
| Package Manager | uv | Fast Python package manager |
| Dataset | climbmix-400b-shuffle | Karpathy's curated training mix |

## What Makes This Unusual

1. **The human programs in Markdown, not Python.** The `program.md` file is the real "code" the human iterates on — it's the instruction set for the AI researcher.

2. **Fixed time budget, not fixed steps.** Training always runs for exactly 5 minutes wall clock. This makes experiments comparable regardless of what the agent changes.

3. **The AI agent never stops.** The protocol explicitly says the agent should run indefinitely. ~100 experiments while the human sleeps.

4. **Single-file scope constraint.** The agent can only modify `train.py`. This is a deliberate design choice to keep complexity manageable and diffs reviewable.
