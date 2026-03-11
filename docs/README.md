# Autoresearch Documentation

Welcome to the **autoresearch** documentation. This project is an autonomous AI research system created by Andrej Karpathy that lets an AI agent (like Claude) independently run experiments on a small LLM training setup — modifying code, training for 5 minutes, evaluating, keeping or discarding changes, and repeating indefinitely.

> *"One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun..."* — @karpathy

## Quick Navigation

| Document | What You'll Learn |
|----------|-------------------|
| [Architecture Overview](architecture-overview.md) | The big picture — how all the pieces fit together |
| [Data Flow](data-flow.md) | How data moves through the system, from download to evaluation |
| [Data Schema](data-schema.md) | File formats, data structures, and storage layout |
| [Model Architecture](model-architecture.md) | The GPT model — every layer, every trick, explained |
| [Optimizer & Training](optimizer.md) | MuonAdamW optimizer, learning rate schedules, training loop |
| [Agent Architecture](agent-architecture.md) | How the AI agent runs autonomous experiments |
| [APIs & Interfaces](apis-and-interfaces.md) | Internal module interfaces and function signatures |
| [Configuration](configuration.md) | Every hyperparameter and knob you can tune |
| [Design Rationale](design-rationale.md) | Why things are built the way they are |

## The Three Files That Matter

The entire project is intentionally tiny — just three files:

1. **`prepare.py`** — Data prep & runtime utilities (read-only, never modified by the agent)
2. **`train.py`** — The model, optimizer, and training loop (the agent modifies this file)
3. **`program.md`** — Instructions for the AI agent (the human modifies this file)

Start with the [Architecture Overview](architecture-overview.md) if you're new, or jump to any section that interests you.
