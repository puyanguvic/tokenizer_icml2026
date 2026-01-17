"""Tokenizer evaluation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctok_core.tokenization.io import load_tokenizer
from ctok_core.tokenization.runtime import TokenizerRuntime


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("eval", help="Evaluate tokenizer length statistics.")
    parser.add_argument("--artifact", required=True, help="Tokenizer artifact directory.")
    parser.add_argument("--corpus", required=True, help="Corpus file to evaluate.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.set_defaults(func=run)
    return parser


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact)
    corpus_path = Path(args.corpus)

    vocab, _rules, _manifest = load_tokenizer(artifact_dir)
    runtime = TokenizerRuntime(vocab)

    lengths: list[int] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            lengths.append(len(runtime.encode(text)))

    if not lengths:
        raise ValueError("Corpus is empty or contains only blank lines.")

    lengths.sort()
    stats = {
        "count": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "p50": _percentile(lengths, 0.50),
        "p95": _percentile(lengths, 0.95),
    }

    output = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0
