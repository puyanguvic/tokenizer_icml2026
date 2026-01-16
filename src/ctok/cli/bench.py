"""Tokenizer benchmarking command."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ctok.tokenization.io import load_tokenizer
from ctok.tokenization.runtime import TokenizerRuntime


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench", help="Benchmark tokenizer throughput.")
    parser.add_argument("--artifact", required=True, help="Tokenizer artifact directory.")
    parser.add_argument("--input", required=True, help="Input text file.")
    parser.add_argument("--repeat", type=int, default=10, help="Repeat count.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    vocab, _rules, _manifest = load_tokenizer(Path(args.artifact))
    runtime = TokenizerRuntime(vocab)

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Input is empty.")

    for _ in range(args.warmup):
        for line in lines:
            runtime.encode(line)

    start = time.perf_counter()
    total_tokens = 0
    for _ in range(args.repeat):
        for line in lines:
            total_tokens += len(runtime.encode(line))
    elapsed = time.perf_counter() - start

    tokens_per_sec = total_tokens / elapsed if elapsed else 0.0
    print(f"tokens_per_sec={tokens_per_sec:.2f}")
    return 0
