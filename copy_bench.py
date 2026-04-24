#!/usr/bin/env python3
"""Verbatim-copy benchmark for long-context LLMs.

Motivation: DeepSeek-V4's hybrid CSA/HCA attention has NO token-level full
attention outside the sliding window. Distant tokens must be reconstructed
from compressed KV summaries (CSA: group of m tokens -> 1 entry; HCA: m' >> m
tokens -> 1 entry). This suite measures the degradation as span length and
context length grow.

Two specification modes
  marker : insert unique [[S_UID]] ... [[E_UID]] markers around the target span
           and ask the model to output exactly what lies between them.
           Pure content copy — no counting, no indexing.
  lines  : number every line (NNNNNN| ...) and ask for lines [L1, L2] inclusive.
           Copy + positional indexing.

Metrics per case: exact match, gold/pred length, Levenshtein distance,
normalized edit distance, first divergence offset, longest-common-substring
ratio. Written as one JSON object per line to the output file.

Dependencies
  pip install datasets openai

Usage
  export DEEPSEEK_API_KEY=sk-...
  python deepseek_v4_copy_test.py \
      --model deepseek-chat \
      --base-url https://api.deepseek.com/v1 \
      --out results.jsonl

Works with any OpenAI-compatible endpoint (vLLM, sglang, TGI, OpenAI, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterable

from openai import OpenAI

# ~4 chars / token for English; used only for rough sizing.
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------- text source


def load_text(min_chars: int, source: str, text_file: str | None = None) -> str:
    """Load a long continuous natural-language text."""
    if text_file:
        with open(text_file) as f:
            t = f.read()
        while len(t) < min_chars:
            t = t + "\n\n" + t
        return t[:min_chars]

    from datasets import load_dataset

    if source == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
        key = "text"
    elif source == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        key = "text"
    elif source == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
        key = "text"
    else:
        raise ValueError(f"unknown source: {source}")

    buf: list[str] = []
    total = 0
    for ex in ds:
        t = ex[key]
        if not t:
            continue
        buf.append(t)
        total += len(t) + 2
        if total >= min_chars:
            break
    text = "\n\n".join(buf)
    return text[:min_chars]


# ---------------------------------------------------------------- prompt build


def _snap_to_word(text: str, pos: int, direction: int) -> int:
    n = len(text)
    pos = max(0, min(n, pos))
    while 0 < pos < n and not text[pos].isspace():
        pos += direction
        if pos <= 0 or pos >= n:
            break
    return pos


def pick_span(text: str, span_chars: int, position: float) -> tuple[int, int]:
    """Pick [start, end) of length ~span_chars, centered at relative `position`."""
    n = len(text)
    span_chars = min(span_chars, n // 2)
    center = int(position * (n - span_chars)) + span_chars // 2
    start = center - span_chars // 2
    end = start + span_chars
    start = _snap_to_word(text, start, -1)
    end = _snap_to_word(text, end, +1)
    return start, end


def build_marker_prompt(text: str, start: int, end: int, uid: str) -> tuple[str, str]:
    gold = text[start:end]
    marked = (
        text[:start]
        + f"[[S_{uid}]]"
        + gold
        + f"[[E_{uid}]]"
        + text[end:]
    )
    prompt = (
        "Below is a long document containing two unique markers "
        f"[[S_{uid}]] and [[E_{uid}]] at some position. "
        "Output the text strictly between these two markers, exactly "
        "character-for-character. Do NOT include the markers themselves. "
        "Do NOT add any commentary, quoting, code fences, or surrounding "
        "whitespace. Output only the raw text.\n\n"
        "---BEGIN DOCUMENT---\n"
        f"{marked}\n"
        "---END DOCUMENT---\n\n"
        f"Now output the exact text between [[S_{uid}]] and [[E_{uid}]]:"
    )
    return prompt, gold


def build_lines_prompt(text: str, line_start: int, line_end: int) -> tuple[str, str]:
    lines = text.splitlines()
    numbered = "\n".join(f"{i+1:06d}| {ln}" for i, ln in enumerate(lines))
    gold = "\n".join(lines[line_start - 1 : line_end])
    prompt = (
        "Below is a numbered document. Each line is prefixed with a 6-digit "
        "line number followed by '| '. Output the exact content of lines "
        f"{line_start} through {line_end} inclusive, WITHOUT the line-number "
        "prefix, WITHOUT any commentary, WITHOUT code fences. Just the "
        "original text joined by newlines.\n\n"
        "---BEGIN DOCUMENT---\n"
        f"{numbered}\n"
        "---END DOCUMENT---\n\n"
        f"Now output lines {line_start}..{line_end}:"
    )
    return prompt, gold


def char_range_to_line_range(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.count("\n", 0, start) + 1
    line_end = text.count("\n", 0, end) + 1
    return line_start, line_end


# ---------------------------------------------------------------- metrics


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def first_divergence(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def lcs_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = SequenceMatcher(None, a, b, autojunk=False)
    return 2 * sum(x.size for x in m.get_matching_blocks()) / (len(a) + len(b))


def evaluate(pred: str, gold: str) -> dict:
    pred_s = pred.strip()
    gold_s = gold.strip()
    denom = max(len(pred_s), len(gold_s), 1)
    d = levenshtein(pred_s, gold_s)
    return dict(
        exact=(pred_s == gold_s),
        gold_len=len(gold_s),
        pred_len=len(pred_s),
        edit_distance=d,
        normalized_edit_distance=d / denom,
        first_divergence=first_divergence(pred_s, gold_s),
        lcs_ratio=lcs_ratio(pred_s, gold_s),
    )


# ---------------------------------------------------------------- runner


@dataclass
class Case:
    mode: str             # "marker" | "lines"
    context_tokens: int
    span_tokens: int
    position: float       # 0.0 .. 1.0
    trial: int


def run_case(client: OpenAI, model: str, text: str, case: Case,
             max_out_tokens: int | None = None) -> dict:
    ctx_chars = case.context_tokens * CHARS_PER_TOKEN
    span_chars = case.span_tokens * CHARS_PER_TOKEN
    sub = text[:ctx_chars]
    start, end = pick_span(sub, span_chars, case.position)

    if case.mode == "marker":
        rng = random.Random(
            f"{case.context_tokens}|{case.span_tokens}|{case.position}|{case.trial}"
        )
        uid = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
        prompt, gold = build_marker_prompt(sub, start, end, uid)
    elif case.mode == "lines":
        l1, l2 = char_range_to_line_range(sub, start, end)
        prompt, gold = build_lines_prompt(sub, l1, l2)
    else:
        raise ValueError(case.mode)

    if max_out_tokens is None:
        max_out_tokens = max(256, (len(gold) // CHARS_PER_TOKEN) + 256)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_out_tokens,
        temperature=0.0,
    )
    pred = resp.choices[0].message.content or ""
    latency = time.time() - t0

    metrics = evaluate(pred, gold)
    return dict(
        **asdict(case),
        **metrics,
        latency_s=latency,
        prompt_chars=len(prompt),
    )


def cases(context_sizes: Iterable[int], span_sizes: Iterable[int],
          positions: Iterable[float], modes: Iterable[str],
          trials: int) -> Iterable[Case]:
    for ctx in context_sizes:
        for sp in span_sizes:
            if sp * CHARS_PER_TOKEN > ctx * CHARS_PER_TOKEN // 2:
                continue
            for pos in positions:
                for mode in modes:
                    for t in range(trials):
                        yield Case(mode=mode, context_tokens=ctx,
                                   span_tokens=sp, position=pos, trial=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--source", default="wikitext",
                    choices=["wikitext", "c4", "fineweb"])
    ap.add_argument("--text-file", default=None,
                    help="local text file; overrides --source")
    ap.add_argument("--out", default="copy_results.jsonl")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--context-sizes", type=int, nargs="+",
                    default=[4_000, 32_000, 128_000, 500_000])
    ap.add_argument("--span-sizes", type=int, nargs="+",
                    default=[20, 100, 500, 2_000])
    ap.add_argument("--positions", type=float, nargs="+",
                    default=[0.05, 0.5, 0.95])
    ap.add_argument("--modes", nargs="+",
                    default=["marker", "lines"],
                    choices=["marker", "lines"])
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"error: set ${args.api_key_env}", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    min_chars = max(args.context_sizes) * CHARS_PER_TOKEN + 1000
    src = args.text_file if args.text_file else args.source
    print(f"loading ~{min_chars:,} chars from {src} ...", file=sys.stderr)
    text = load_text(min_chars, args.source, text_file=args.text_file)
    print(f"loaded {len(text):,} chars", file=sys.stderr)

    with open(args.out, "w") as fout:
        for case in cases(args.context_sizes, args.span_sizes,
                          args.positions, args.modes, args.trials):
            try:
                row = run_case(client, args.model, text, case)
            except Exception as e:
                row = dict(**asdict(case), error=repr(e))
                print(f"  {case} ERROR {e}", file=sys.stderr)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if "error" not in row:
                print(
                    f"  ctx={case.context_tokens:>7} span={case.span_tokens:>5} "
                    f"pos={case.position:<4} mode={case.mode:<6} "
                    f"t={case.trial} exact={row['exact']!s:<5} "
                    f"ned={row['normalized_edit_distance']:.3f} "
                    f"lcs={row['lcs_ratio']:.3f} "
                    f"first_div={row['first_divergence']}/{row['gold_len']}"
                )

    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
