# Design Rationale

This document explains **why** things are built the way they are. Every design choice in autoresearch is deliberate.

## 1. "Why Only Three Files?"

**Decision:** The entire project is just `prepare.py`, `train.py`, and `program.md`.

**Rationale:**
- **For the AI agent:** Fewer files = less context needed. The agent can read the entire codebase in seconds and understand everything. Complex multi-file projects confuse agents (import chains, config systems, abstractions).
- **For the human:** The entire experiment is one file (`train.py`). You can `git diff` a single file to see exactly what changed in any experiment.
- **For reproducibility:** Copy `train.py` from any commit and you have the complete experiment definition. No hidden configs, no external state.

**The alternative (and why it was rejected):** A typical ML project might have `model.py`, `optimizer.py`, `config.yaml`, `train.py`, `utils.py`, `data.py`, etc. This is better software engineering, but worse for autonomous agents. The agent would need to coordinate changes across multiple files, handle imports correctly, and keep configs in sync. A single file eliminates all of this.

## 2. "Why a Fixed 5-Minute Time Budget?"

**Decision:** Training always runs for exactly 5 minutes wall clock time (excluding startup/compilation).

**Rationale:**
- **Comparability:** Every experiment uses the same amount of compute. Whether the agent doubles the model size (fewer steps) or halves it (more steps), the comparison is fair — "what's the best val_bpb in 5 minutes?"
- **Throughput:** 5 minutes gives ~12 experiments/hour, or ~100 overnight. Short enough for rapid iteration, long enough for meaningful training.
- **Hardware-specific optimization:** The agent optimizes for YOUR GPU. An H100 and an RTX 4090 will get different results because they can do different amounts of work in 5 minutes. This is a feature — you're finding the best model for your hardware.
- **No hyperparameter inflation:** Without a time budget, the agent could "improve" by just training longer. The fixed budget forces actual algorithmic improvements.

**The downside:** Results aren't comparable across different hardware. Your H100 results won't match someone's RTX 3090 results. This is an accepted tradeoff.

## 3. "Why Does the Agent Only Edit train.py?"

**Decision:** The agent can only modify `train.py`. `prepare.py` is read-only.

**Rationale:**
- **Metric integrity:** If the agent could modify `evaluate_bpb`, it could "improve" val_bpb by changing how evaluation works rather than improving the model. The read-only `prepare.py` is a trust boundary.
- **Data integrity:** The dataloader and tokenizer must remain consistent across experiments. If the agent changed tokenization mid-experiment, results would be meaningless.
- **Scope control:** Limiting the agent to one file keeps experiments manageable. Diffs are small, changes are reviewable, and rollbacks are clean.
- **Error containment:** If the agent introduces a bug, it's in `train.py`. A `git reset` fixes everything. If it could modify data loading code, a bug might corrupt cached data.

## 4. "Why Markdown for Agent Instructions?"

**Decision:** Agent instructions are in `program.md`, a Markdown file, not code.

**Rationale:**
- **Agent-agnostic:** Any AI agent (Claude, Codex, GPT, etc.) can read Markdown. No framework lock-in, no API-specific code.
- **Human-readable:** The research protocol is written in plain English. Anyone can read it and understand what the agent does.
- **Iteration-friendly:** The human iterates on `program.md` like they'd iterate on a prompt. No redeployment, no code changes — just edit the text.
- **The insight:** The "code" the human writes isn't Python — it's the research protocol in natural language. The human is "programming the researcher," not programming the model.

## 5. "Why BPB Instead of Perplexity?"

**Decision:** The evaluation metric is bits per byte (BPB), not perplexity or cross-entropy loss.

**Rationale:**
- **Vocab-size independence:** Perplexity depends on vocabulary size. If the agent experiments with different vocab sizes (or the tokenizer changes), perplexity values are incomparable. BPB normalizes by the number of bytes, not tokens.
- **Interpretable:** BPB measures "how many bits does the model need per byte of text?" This has a clear information-theoretic meaning. Lower BPB = better compression = better language model.
- **Standard:** BPB is becoming the standard metric in modern LLM evaluation (used by Chinchilla, PaLM, etc.).

**How it works:** Sum cross-entropy losses (in nats), sum byte lengths of target tokens, divide nats by (ln(2) × bytes). Special tokens are excluded.

## 6. "Why Muon + AdamW Hybrid?"

**Decision:** Use Muon optimizer for 2D weight matrices, AdamW for everything else.

**Rationale:**
- **Muon's strength:** Muon (based on orthogonal gradient projection) has been shown to significantly outperform Adam for weight matrix training, especially in transformer architectures. It normalizes the update direction, preventing some layers from dominating training.
- **Muon's weakness:** Muon requires matrix-shaped parameters for polar decomposition. It doesn't work for 1D vectors (biases), scalars, or embedding tables.
- **AdamW fills the gap:** Embeddings, the output head, and per-layer scalars all use standard AdamW with per-group learning rates.
- **Best of both worlds:** You get Muon's superior matrix optimization plus Adam's reliable handling of everything else.

## 7. "Why Value Embeddings (ResFormer)?"

**Decision:** Alternating layers have separate value embedding tables that mix into the attention values.

**Rationale:**
- **The problem:** In standard transformers, the attention value (V) is derived entirely from the current hidden state. But the hidden state has been through multiple layers of transformation, potentially losing direct information about the input tokens.
- **The solution:** Value embeddings provide a "shortcut" — the model can look up a learned value vector directly from the input token ID, bypassing all intermediate layers.
- **Why alternating:** Every-layer value embeddings would nearly double the parameter count (each is `vocab_size × kv_dim`). Alternating provides the benefit at half the cost.
- **The gating mechanism:** A small gate network decides how much value embedding to mix in, per head. Initialized to neutral (1.0), so it starts as a simple residual and the model learns when to use it.

## 8. "Why Differential Residual Connections?"

**Decision:** Each block mixes `resid_lambdas[i] * x + x0_lambdas[i] * x0` instead of standard `x + block(x)`.

**Rationale:**
- **Gradient flow:** In very deep networks, gradients must flow through many layers. The x0 shortcut provides a direct path from any layer back to the input, similar to DenseNet-style connections but much cheaper.
- **Learnable mixing:** The model learns per-layer how much to rely on the accumulated representation (x) vs. the raw input (x0). Early layers might rely more on x0, later layers more on x.
- **Initialization:** `resid_lambdas = 1.0`, `x0_lambdas = 0.1` — the model starts as a standard residual network with a small x0 perturbation, and learns to adjust from there.

## 9. "Why No Config Files?"

**Decision:** All hyperparameters are Python constants at the top of `train.py`, not in YAML/JSON/TOML config files.

**Rationale:**
- **Agent-friendly:** The agent edits code, not config files. Having constants directly in the code means the agent can see and change them in the same place as the rest of the logic.
- **No parsing overhead:** No config loading, no validation, no default merging. Just Python variables.
- **Self-documenting:** The constants are right there in the file with inline comments. No need to cross-reference a config schema.
- **Diff-friendly:** `git diff` shows hyperparameter changes alongside code changes, giving full context.

## 10. "Why torch.compile?"

**Decision:** The entire model is compiled with `torch.compile(model, dynamic=False)`, and optimizer steps use `@torch.compile`.

**Rationale:**
- **Performance:** `torch.compile` fuses operations, eliminates Python overhead, and generates optimized CUDA kernels. Typical speedup: 1.5-2× for transformer training.
- **`dynamic=False`:** Since batch size and sequence length are fixed, there's no need for dynamic shape support. This allows more aggressive compilation.
- **`fullgraph=True` for optimizers:** The optimizer steps are fully compiled as single graphs, maximizing fusion opportunities.
- **Warmup exclusion:** The first ~10 steps are compilation warmup (slow). The time budget starts counting after these steps, so compilation doesn't penalize the experiment.

## 11. "Why Pin the Validation Shard?"

**Decision:** Shard 6542 (the last shard) is always the validation shard.

**Rationale:**
- **Consistency:** Every experiment evaluates on exactly the same data. No random validation splits that might vary.
- **No overlap:** By using the last shard and training on earlier shards, there's zero overlap regardless of how many training shards you download.
- **Simplicity:** No complex train/val splitting logic. The validation shard is just a constant.

## 12. "Why Best-Fit Packing?"

**Decision:** The dataloader uses best-fit bin packing to fill fixed-size rows with variable-length documents.

**Rationale:**
- **Zero waste:** Every position in every row contains a real token. No padding tokens wasted.
- **Document integrity:** Most documents are packed whole (not truncated). Only when no document fits the remaining space is the shortest document cropped.
- **Better than alternatives:**
  - Simple concatenation + chunking: Breaks document boundaries, no BOS tokens
  - Padding to max length: Wastes tokens on padding
  - Truncation: Loses information from long documents
- **BOS alignment:** Every document starts with a BOS token, giving the model clean document boundaries.

## 13. "Why the 'Never Stop' Directive?"

**Decision:** `program.md` explicitly tells the agent to never stop or ask for confirmation.

**Rationale:**
- **Overnight operation:** The primary use case is running while the human sleeps. Any "should I continue?" prompt would halt the entire system.
- **Autonomous research:** The whole point is autonomous operation. Stopping to ask defeats the purpose.
- **Resource utilization:** GPU time is expensive. Every minute spent waiting for human input is a wasted experiment.

## 14. "Why No Distributed Training?"

**Decision:** Single GPU only. No data parallel, no model parallel, no FSDP.

**Rationale:**
- **Simplicity:** Distributed training adds enormous complexity (communication, synchronization, sharding strategies). This complexity would make the codebase harder for the agent to understand and modify.
- **Single-file constraint:** Distributed training typically requires infrastructure code. Keeping everything in one file rules this out.
- **Sufficient for the use case:** A single H100 can train a meaningful model in 5 minutes. The insight is that you don't need huge models — you need rapid iteration.

## 15. "Why GC Management?"

**Decision:** Python's garbage collector is disabled after the first step and only run every 5000 steps.

**Rationale:**
- **Consistent timing:** Python's GC causes unpredictable 500ms stalls. In a 5-minute training run, each stall wastes ~0.17% of the budget.
- **GC is unnecessary:** During training, memory allocation patterns are predictable. Objects created during a step are freed at the end of the step. The reference-counting GC handles this. The cyclic GC (which causes stalls) isn't needed.
- **`gc.freeze()`:** Tells the GC to not scan objects that exist at freeze time. This means the model parameters (which never become garbage) aren't repeatedly scanned.

## Summary of Design Philosophy

The entire project embodies a few key principles:

1. **Simplicity over elegance** — One file, no abstractions, no configs
2. **Agent-first design** — Everything is optimized for AI agent comprehension and modification
3. **Fair comparison** — Fixed time budget, fixed evaluation, fixed data
4. **Safety boundaries** — Read-only evaluation prevents gaming
5. **Autonomous operation** — Designed to run overnight without human intervention
6. **Hardware-aware** — Optimizes for YOUR specific GPU, not an abstract benchmark
