"""Tokenizer build command."""

from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path

import yaml

from ctok.induction.candidates import collect_ngrams, collect_ngrams_with_labels
from ctok.induction.distortion import NullDistortion, build_label_entropy_distortion
from ctok.induction.greedy import greedy_select
from ctok.tokenization.boundary import DEFAULT_BOUNDARY_CHARS, normalize_boundary_chars
from ctok.tokenization.rules import RuleSet
from ctok.tokenization.tokenizer import CtokTokenizer
from ctok.tokenization.vocab import Vocabulary


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("build", help="Build a tokenizer artifact.")
    parser.add_argument("--corpus", required=True, help="Path to raw text corpus.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument("--labels", help="Optional label file aligned with corpus.")
    parser.add_argument("--config", help="Tokenizer config YAML.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Total vocab size.")
    parser.add_argument("--min-freq", type=int, default=None, help="Minimum ngram frequency.")
    parser.add_argument("--min-len", type=int, default=None, help="Minimum ngram length.")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum ngram length.")
    parser.add_argument("--lambda-weight", type=float, default=None, help="Distortion weight.")
    parser.add_argument("--max-lines", type=int, default=None, help="Max lines to read (0 = all).")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token mapping name=token (repeatable).",
    )
    parser.add_argument(
        "--boundary-aware",
        action="store_true",
        default=None,
        help="Restrict candidates to boundary-respecting spans.",
    )
    parser.add_argument(
        "--boundary-chars",
        default=None,
        help="Boundary characters string used for boundary-aware candidates.",
    )
    parser.set_defaults(func=run)
    return parser


def _load_lines(path: Path, max_lines: int) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            lines.append(line.rstrip("\n"))
            if max_lines and idx + 1 >= max_lines:
                break
    return lines


def _collect_charset(lines: list[str]) -> list[str]:
    charset: set[str] = set()
    for line in lines:
        charset.update(line)
    return sorted(charset)


def run(args: argparse.Namespace) -> int:
    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)

    config = _load_config(args.config)
    config_values = _extract_tokenizer_config(config)

    vocab_size = _resolve_arg(args.vocab_size, config_values.get("vocab_size"), 2048)
    min_freq = _resolve_arg(args.min_freq, config_values.get("min_freq"), 2)
    min_len = _resolve_arg(args.min_len, config_values.get("min_len"), 2)
    max_len = _resolve_arg(args.max_len, config_values.get("max_len"), 8)
    lambda_weight = _resolve_arg(args.lambda_weight, config_values.get("lambda_weight"), 0.0)
    max_lines = _resolve_arg(args.max_lines, config_values.get("max_lines"), 0)

    boundary_aware = _resolve_arg(
        args.boundary_aware,
        _resolve_boundary_aware(config_values),
        False,
    )
    boundary_chars = _resolve_boundary_chars(args.boundary_chars, config_values.get("boundary_chars"))
    if not boundary_aware:
        boundary_chars = set()

    lines = _load_lines(corpus_path, max_lines)
    if not lines:
        raise ValueError("Corpus is empty.")

    base_charset = str(config_values.get("base_charset", "corpus")).lower()
    if base_charset == "byte":
        base_vocab = [chr(i) for i in range(256)]
    else:
        base_vocab = _collect_charset(lines)
    if boundary_aware and boundary_chars:
        for ch in sorted(boundary_chars):
            if ch not in base_vocab:
                base_vocab.append(ch)
    if len(base_vocab) >= vocab_size:
        raise ValueError("Base charset exceeds requested vocab size.")

    distortion = NullDistortion()
    if args.labels:
        labels = _load_labels(Path(args.labels), max_lines)
        if len(labels) != len(lines):
            raise ValueError("Label file length does not match corpus lines.")
        candidates, label_counts = collect_ngrams_with_labels(
            lines,
            labels,
            min_len=min_len,
            max_len=max_len,
            min_freq=min_freq,
            boundary_chars=boundary_chars if boundary_aware else None,
        )
        distortion = build_label_entropy_distortion(label_counts)
    else:
        candidates = collect_ngrams(
            lines,
            min_len=min_len,
            max_len=max_len,
            min_freq=min_freq,
            boundary_chars=boundary_chars if boundary_aware else None,
        )

    selected = greedy_select(
        candidates=candidates,
        budget=vocab_size - len(base_vocab),
        lambda_weight=lambda_weight,
        distortion=distortion,
    )

    special_tokens = _parse_special_tokens(args.special_token)
    config_special = config_values.get("special_tokens") or {}
    if isinstance(config_special, dict):
        for name, token in config_special.items():
            if name not in special_tokens:
                special_tokens[name] = token
    vocab = Vocabulary(tokens=_merge_tokens(base_vocab + selected, special_tokens), special_tokens=special_tokens)
    rules = RuleSet.from_vocab(vocab)
    metadata = {
        "corpus": str(corpus_path),
        "vocab_size": vocab_size,
        "min_freq": min_freq,
        "min_len": min_len,
        "max_len": max_len,
        "lambda_weight": lambda_weight,
        "max_lines": max_lines,
        "special_tokens": special_tokens,
        "boundary_aware": boundary_aware,
        "boundary_chars": sorted(boundary_chars) if boundary_chars else None,
        "labels_path": str(args.labels) if args.labels else None,
    }
    tokenizer = CtokTokenizer(
        vocab=vocab,
        rules=rules,
        special_tokens=special_tokens,
        boundary_mode="aware" if boundary_aware else "none",
        boundary_chars=sorted(boundary_chars) if boundary_chars else None,
    )
    tokenizer.save_pretrained(output_dir, metadata=metadata)
    return 0


def _parse_special_tokens(entries: list[str]) -> dict[str, str]:
    special: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid special token '{entry}'. Use name=token.")
        name, token = entry.split("=", 1)
        name = name.strip()
        token = token.strip()
        if not name or not token:
            raise ValueError(f"Invalid special token '{entry}'. Use name=token.")
        special[name] = token
    return special


def _merge_tokens(tokens: list[str], special_tokens: dict[str, str]) -> list[str]:
    merged = list(tokens)
    for token in special_tokens.values():
        if token not in merged:
            merged.append(token)
    return merged


def _load_labels(path: Path, max_lines: int) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            labels.append(line.rstrip("\n"))
            if max_lines and idx + 1 >= max_lines:
                break
    if not labels:
        raise ValueError("Label file is empty.")
    return labels


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer config must be a mapping.")
    return payload


def _extract_tokenizer_config(payload: dict[str, Any]) -> dict[str, Any]:
    if "tokenizer" in payload and isinstance(payload["tokenizer"], dict):
        return payload["tokenizer"]
    return payload


def _resolve_arg(value: Any, config_value: Any, default: Any) -> Any:
    if value is not None:
        return value
    if config_value is not None:
        return config_value
    return default


def _resolve_boundary_aware(config_values: dict[str, Any]) -> bool | None:
    boundary_mode = config_values.get("boundary_mode")
    if boundary_mode is not None:
        return boundary_mode == "aware"
    return config_values.get("boundary_aware")


def _resolve_boundary_chars(cli_value: str | None, config_value: Any) -> set[str]:
    if cli_value is not None:
        return normalize_boundary_chars(cli_value)
    if config_value is not None:
        return normalize_boundary_chars(config_value)
    return DEFAULT_BOUNDARY_CHARS
