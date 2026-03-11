# Agent Architecture

This document explains how the AI agent operates as an autonomous researcher, including the experiment protocol, branching strategy, and decision-making process.

## Overview

The "agent" in autoresearch is not a special piece of software — it's a general-purpose AI coding assistant (like Claude, Codex, or similar) that reads `program.md` and follows its instructions. The brilliance of this design is that the "agent architecture" is just a Markdown file.

```
┌────────────────────────────────────────────────────────────┐
│                    AI Coding Agent                          │
│                 (Claude, Codex, etc.)                       │
│                                                            │
│  Reads: program.md (instructions)                          │
│         README.md (context)                                │
│         prepare.py (constants and constraints)             │
│         train.py (current state of experiment)             │
│                                                            │
│  Writes: train.py (modified code)                          │
│          results.tsv (experiment log)                      │
│          run.log (training output)                         │
│                                                            │
│  Runs: git commands, uv run train.py, grep                 │
└────────────────────────────────────────────────────────────┘
```

## The Agent Protocol (from program.md)

The agent follows a two-phase protocol:

### Phase 1: Setup

1. **Agree on a run tag** with the human (e.g., `mar5`)
2. **Create a branch**: `git checkout -b autoresearch/<tag>`
3. **Read all files** for full context (`README.md`, `prepare.py`, `train.py`)
4. **Verify data** exists in `~/.cache/autoresearch/`
5. **Initialize** `results.tsv` with a header row
6. **Run baseline**: Execute unmodified `train.py` to establish the starting `val_bpb`

### Phase 2: Experiment Loop (Infinite)

```
┌─────────────────────────────────────────────────────┐
│                  EXPERIMENT LOOP                     │
│                                                      │
│  1. Think of an experiment idea                      │
│  2. Edit train.py                                    │
│  3. git commit                                       │
│  4. Run: uv run train.py > run.log 2>&1              │
│  5. Parse results: grep "^val_bpb:" run.log          │
│  6. If crash → read error, maybe fix, log as crash   │
│  7. Log to results.tsv                               │
│  8. If val_bpb improved → KEEP (advance branch)      │
│     If val_bpb same/worse → DISCARD (git reset)      │
│  9. Go to step 1                                     │
│                                                      │
│  NEVER STOP. Run until human interrupts.             │
└─────────────────────────────────────────────────────┘
```

## Decision Logic

### Keep vs. Discard

The decision is simple:

```
if new_val_bpb < best_val_bpb:
    status = "keep"       # advance the branch
else:
    status = "discard"    # git reset to previous commit
```

There's also a **simplicity criterion**: all else being equal, simpler is better.

| Scenario | Decision |
|----------|----------|
| 0.001 improvement + 20 lines of hacky code | Probably not worth it |
| 0.001 improvement from deleting code | Definitely keep |
| ~0 improvement but much simpler code | Keep |
| Any improvement from clean, simple change | Keep |

### Crash Handling

When a run crashes:

1. Run `tail -n 50 run.log` to read the error
2. If it's a simple bug (typo, missing import): fix and re-run
3. If it's a fundamental problem (OOM, incompatible architecture): log as crash, revert, move on
4. If stuck after a few attempts: give up on that idea

### Timeout Handling

- Normal run: ~5 minutes (+ startup overhead)
- If a run exceeds 10 minutes: kill it, treat as failure, discard and revert

## Git Branching Strategy

```
main
  │
  └── autoresearch/mar5  (experiment branch)
       │
       ├── commit: baseline
       ├── commit: increase LR (keep ✓)
       ├── commit: try GeLU (discard ✗ → reverted)
       ├── commit: wider model (crash ✗ → reverted)
       ├── commit: lower depth (keep ✓)
       └── ... (continues indefinitely)
```

**Key points:**
- Each experiment is a single branch
- Kept experiments advance the branch (git commit stays)
- Discarded experiments are reverted (`git reset`)
- The branch tells the linear story of improvements
- `results.tsv` is **not** committed (untracked)

**Rationale:** This branching strategy means you can `git log` the experiment branch and see only the successful improvements. The TSV file keeps the full history (including failures) for analysis.

## What the Agent Can and Cannot Do

### CAN Do (Fair Game)

Everything in `train.py`:
- Model architecture (layers, dimensions, heads, attention patterns)
- Optimizer settings (learning rates, betas, weight decay)
- Training hyperparameters (batch size, gradient accumulation)
- Activation functions
- Initialization strategies
- Any creative architectural change

### CANNOT Do (Off Limits)

- Modify `prepare.py` (data loading, tokenizer, evaluation)
- Install new packages or dependencies
- Modify the evaluation metric (`evaluate_bpb`)
- Modify the time budget or sequence length constants

**Rationale:** The constraints create a fair playing field. By fixing the evaluation and data pipeline, every experiment is comparable. The agent can't game the metric or change the rules — only improve the model and training procedure.

## Why This Design Works

### 1. Markdown as "Agent Code"

The `program.md` file is essentially the "source code" for the research organization. But it's written in natural language Markdown, not Python. This means:
- **Any human can understand** the research protocol
- **Any AI agent can follow** the instructions (not locked to a specific agent framework)
- **Easy to iterate**: change the instructions, not the code
- **Composable**: you could add more agents with different `program.md` files

### 2. Fixed Time Budget as Comparability

By fixing training to 5 minutes wall clock:
- Every experiment is directly comparable regardless of what the agent changes
- The agent can't "cheat" by training longer
- Results are specific to your hardware (feature, not bug — you're optimizing for YOUR GPU)
- ~12 experiments/hour, ~100 overnight

### 3. Single-File Constraint

Restricting the agent to only modify `train.py`:
- Keeps diffs small and reviewable
- Prevents the agent from accidentally breaking infrastructure
- Makes experiments reproducible (the full experiment is one file)
- Forces creativity within constraints

### 4. "Never Stop" Directive

The protocol explicitly states:
> *"Once the experiment loop has begun, do NOT pause to ask the human if you should continue... The human might be asleep."*

This ensures the agent runs autonomously through the night. Expected throughput:
- ~5 minutes per experiment
- ~12 experiments per hour
- ~100 experiments over ~8 hours of sleep

## Agent Capabilities & Ideas

The program tells the agent to try a wide range of ideas:
- Hyperparameter tuning (LR, batch size, depth, width)
- Architecture changes (activation functions, attention variants)
- Optimizer modifications
- Training tricks (gradient clipping, regularization)
- If stuck: "think harder — read papers referenced in the code, re-read the in-scope files, try combining near-misses, try radical changes"

## The Analysis Notebook

After experiments complete, `analysis.ipynb` provides visualization:
- Loads `results.tsv`
- Plots val_bpb progression over experiments
- Shows keep/discard/crash statistics
- Highlights the best-performing experiments
- Shows memory usage trends

**Rationale:** The notebook gives the human a quick overview when they wake up: "What happened while I was asleep?"
