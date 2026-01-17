"""Tokenizer encode command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ctok_core.tokenization.io import load_tokenizer
from ctok_core.tokenization.runtime import TokenizerRuntime


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("encode", help="Encode inputs using a tokenizer artifact.")
    parser.add_argument("--artifact", required=True, help="Tokenizer artifact directory.")
    parser.add_argument("--input", help="Input text file (defaults to stdin).")
    parser.add_argument("--output", help="Output file (defaults to stdout).")
    parser.add_argument(
        "--format",
        default="ids",
        choices=["ids", "tokens"],
        help="Output format.",
    )
    parser.set_defaults(func=run)
    return parser


def _read_text(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def _write_text(path: Path | None, text: str) -> None:
    if path is None:
        sys.stdout.write(text)
        return
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact)
    vocab, _rules, _manifest = load_tokenizer(artifact_dir)
    runtime = TokenizerRuntime(vocab)

    text = _read_text(Path(args.input) if args.input else None)
    if not text:
        _write_text(Path(args.output) if args.output else None, "")
        return 0

    output_lines: list[str] = []
    for line in text.splitlines():
        ids = runtime.encode(line)
        if args.format == "tokens":
            tokens = [vocab.token_for(i) for i in ids]
            output_lines.append(" ".join(tokens))
        else:
            output_lines.append(" ".join(str(i) for i in ids))

    output = "\n".join(output_lines) + ("\n" if output_lines else "")
    _write_text(Path(args.output) if args.output else None, output)
    return 0
