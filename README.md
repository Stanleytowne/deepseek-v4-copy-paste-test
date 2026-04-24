# deepseek-v4-copy-test

Benchmark for verbatim copy / strict recall on long-context LLMs. Designed to
stress-test models whose attention has no token-level full attention over the
full context — e.g. DeepSeek-V4's hybrid **CSA** (Compressed Sparse Attention)
and **HCA** (Heavily Compressed Attention).

## Motivation

Per the DeepSeek-V4 tech report §2.3, every attention layer is one of:

- **CSA** — compresses every `m` tokens' KV into one entry, uses a Lightning
  Indexer to pick top-k compressed blocks, and attends over those compressed
  blocks plus a small sliding window of raw recent tokens.
- **HCA** — compresses every `m' (≫ m)` tokens into one entry, dense attention
  over all compressed entries plus a sliding window.

The only place raw token-level KV is accessed is the per-layer sliding window.
Every distant token must be reconstructed from a compressed vector that
summarises `m` (CSA) or `m' ≫ m` (HCA) tokens. Strict verbatim copy of distant
spans is therefore an information-theoretic challenge, not just a recall one.

This benchmark measures how copy accuracy degrades as span length, context
length, and span position vary.

## Setup

```
pip install datasets openai
export DEEPSEEK_API_KEY=sk-...
```

Works with any OpenAI-compatible endpoint (DeepSeek API, vLLM, SGLang, TGI,
OpenAI, etc.).

## Usage

```
python copy_bench.py \
    --model deepseek-v4-pro \
    --base-url https://api.deepseek.com/v1 \
    --out v4pro.jsonl
```

Useful flags:

| flag | meaning | default |
|---|---|---|
| `--source` | `pg19` / `wikipedia` / `c4` | `pg19` |
| `--context-sizes` | context lengths in tokens | `4000 32000 128000 500000` |
| `--span-sizes` | target span lengths in tokens | `20 100 500 2000` |
| `--positions` | where the span sits in `[0,1]` | `0.05 0.5 0.95` |
| `--modes` | `marker` and/or `lines` | both |
| `--trials` | repetitions per config | `3` |

## Spec modes

- **marker** — wraps the target range with `[[S_UID]]` / `[[E_UID]]` and asks
  the model to output what lies strictly between them. Pure copy, no counting.
- **lines** — prefixes every line with `NNNNNN| ` and asks for lines
  `L1..L2`. Copy plus positional indexing.

Running both separates "can it locate the span" from "can it copy it exactly":
the Lightning Indexer should handle location; the bottleneck is copy.

## Metrics (per case)

- `exact` — strict string equality after trim
- `gold_len`, `pred_len` — character counts
- `edit_distance` — Levenshtein
- `normalized_edit_distance`
- `first_divergence` — offset of first mismatching character
- `lcs_ratio` — longest-common-substring similarity
- `latency_s`

Output is one JSON object per line to `--out`.

## Expected failure mode

Compressed-KV reconstruction typically produces *"first N tokens correct,
then drift"*, so `first_divergence ≪ gold_len` is the tell. Accuracy should
degrade monotonically as span length and total context grow, and spans
landing outside the sliding window should do much worse than those inside it.

For a comparison baseline, run the same config against a full-attention model
(Claude, GPT, Gemini, Qwen-dense, etc.) and diff the `normalized_edit_distance`
curves.

## Possible extensions

- **random-string span** — replace the target span with random letters to
  strip n-gram priors, forcing the model to rely purely on KV-stored identity.
- **multi-span** — request several distant spans at once to probe interference.
