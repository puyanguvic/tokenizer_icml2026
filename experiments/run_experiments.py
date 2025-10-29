from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dst.tokenizer import DSTTokenizer

from .metrics import (
    compression_ratio_vs_chars,
    round_trip_accuracy,
    throughput,
    tokens_per_sequence,
)


# Simple domain profiles with regex patterns guiding vocabulary induction
PROTOCOL_PATTERNS = [
    r"https?://[^\s]+",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",  # IPv4
    r"[A-Za-z_][A-Za-z0-9_]*",
    r"[A-Za-z0-9_\-]+=[A-Za-z0-9_\-]+",
    r"[A-Za-z0-9_\-]+\.[a-z]{2,6}",
]

CONFIG_PATTERNS = [
    r"[A-Za-z0-9_\-]+\s*:\s*[^\n#]+",  # YAML-like key: value
    r"-\s+[^\s]+",  # list entries
    r"\{[^{}]+\}",  # inline JSON
    r"\[[^\[\]]+\]",  # inline arrays
    r"\$\{[A-Za-z0-9_\.\-:]+\}",  # placeholders
]

CODE_PATTERNS = [
    r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
    r"class\s+[A-Za-z_][A-Za-z0-9_]*",
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*",
    r"[A-Za-z_][A-Za-z0-9_]*\s*\(",
    r"import\s+[A-Za-z0-9_\.,\s]+",
    r"\"[^\"]+\"|\'[^\']+\'",
]

BIO_PATTERNS = [
    r">[^\n]+",  # FASTA headers
    r"[ACGTN]{6,}",  # nucleotide runs
]


DOMAIN_PATTERNS: Dict[str, Sequence[str]] = {
    "protocol": PROTOCOL_PATTERNS,
    "config": CONFIG_PATTERNS,
    "code": CODE_PATTERNS,
    "bio": BIO_PATTERNS,
    "generic": [],  # fall back to defaults in dst.vocab
}


# Baseline tokenizers implemented locally (no external deps)
class ByteTokenizer:
    def encode(self, text: str) -> List[str]:
        return list(text)

    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)


class WhitespaceTokenizer:
    def encode(self, text: str) -> List[str]:
        return text.split()

    def decode(self, tokens: List[str]) -> str:
        # Loses original whitespace → not invertible in general
        return " ".join(tokens)


def avg_tokens_baseline(tokenizer, corpus: Iterable[str]) -> float:
    total_tokens = 0
    total = 0
    for line in corpus:
        total += 1
        total_tokens += len(tokenizer.encode(line))
    return (total_tokens / total) if total else 0.0


def load_corpus(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    if path.suffix.lower() in {".txt", ".log"}:
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    elif path.suffix.lower() in {".jsonl", ".json"}:
        rows: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                text = obj.get("text") or obj.get("content") or obj.get("value")
                if text is not None:
                    rows.append(str(text))
        return rows
    else:
        raise ValueError(f"Unsupported corpus format: {path.suffix}")


def run(args) -> dict:
    corpus = load_corpus(args.corpus)
    if args.limit and args.limit > 0:
        corpus = corpus[: args.limit]
    if not corpus:
        raise ValueError("Empty corpus after filtering. Provide a non-empty dataset.")

    patterns = DOMAIN_PATTERNS.get(args.domain, []) or None
    tokenizer = DSTTokenizer.train(
        corpus,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
        patterns=patterns,
    )

    # Metrics for DST
    avg_toks = tokens_per_sequence(tokenizer, corpus)
    comp_vs_chars = compression_ratio_vs_chars(tokenizer, corpus)
    rt_acc = round_trip_accuracy(tokenizer, corpus)
    thr = throughput(tokenizer, corpus, trials=args.trials, warmup=args.warmup)

    results = {
        "domain": args.domain,
        "num_samples": len(corpus),
        "dst": {
            "avg_tokens_per_seq": avg_toks,
            "compression_ratio_vs_chars": comp_vs_chars,
            "round_trip_accuracy": rt_acc,
            "throughput": thr,
        },
        "baselines": {},
        "config": {
            "min_freq": args.min_freq,
            "max_vocab": args.max_vocab,
            "patterns": patterns,
        },
    }

    # Baselines
    if not args.no_baselines:
        byte_tok = ByteTokenizer()
        ws_tok = WhitespaceTokenizer()
        results["baselines"] = {
            "byte": {
                "avg_tokens_per_seq": avg_tokens_baseline(byte_tok, corpus),
                "invertible": True,
            },
            "whitespace": {
                "avg_tokens_per_seq": avg_tokens_baseline(ws_tok, corpus),
                "invertible": False,
            },
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce DST experimental metrics on a provided corpus with domain patterns."
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Path to input corpus (.txt or .jsonl)")
    parser.add_argument(
        "--domain",
        choices=list(DOMAIN_PATTERNS.keys()),
        default="generic",
        help="Domain profile controlling regex patterns.",
    )
    parser.add_argument("--min-freq", type=int, default=5, help="Minimum frequency for candidate tokens")
    parser.add_argument("--max-vocab", type=int, default=32000, help="Maximum vocabulary size")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = no limit)")
    parser.add_argument("--trials", type=int, default=1, help="Trials for throughput measurement")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs before timing")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline measurements")
    parser.add_argument("--output", type=Path, help="Write results JSON to this path")

    args = parser.parse_args()
    results = run(args)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

