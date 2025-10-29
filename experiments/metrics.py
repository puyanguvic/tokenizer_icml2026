from __future__ import annotations

import time
from typing import Iterable, List

from dst.tokenizer import DSTTokenizer


def tokens_per_sequence(tokenizer: DSTTokenizer, corpus: Iterable[str]) -> float:
    total_tokens = 0
    total_sequences = 0
    for line in corpus:
        total_sequences += 1
        total_tokens += len(tokenizer.encode(line))
    return (total_tokens / total_sequences) if total_sequences else 0.0


def compression_ratio_vs_chars(tokenizer: DSTTokenizer, corpus: Iterable[str]) -> float:
    total_tokens = 0
    total_chars = 0
    for line in corpus:
        total_tokens += len(tokenizer.encode(line))
        total_chars += len(line)
    return (total_tokens / total_chars) if total_chars else 0.0


def round_trip_accuracy(tokenizer: DSTTokenizer, corpus: Iterable[str]) -> float:
    total = 0
    ok = 0
    for line in corpus:
        total += 1
        if tokenizer.decode(tokenizer.encode(line)) == line:
            ok += 1
    return (ok / total) if total else 1.0


def throughput(tokenizer: DSTTokenizer, corpus: List[str], trials: int = 1, warmup: int = 0) -> dict:
    # Warmup runs (ignored timing) to stabilize JIT/CPU caches
    for _ in range(max(0, warmup)):
        for line in corpus:
            tokenizer.encode(line)

    bytes_processed = sum(len(s.encode("utf-8")) for s in corpus)
    tokens_emitted = 0
    elapsed = 0.0
    for _ in range(max(1, trials)):
        start = time.perf_counter()
        for line in corpus:
            tokens = tokenizer.encode(line)
            tokens_emitted += len(tokens)
        elapsed += (time.perf_counter() - start)

    elapsed = max(elapsed, 1e-9)
    mb_per_s = (bytes_processed / (1024 * 1024)) / (elapsed / max(1, trials))
    tok_per_s = (tokens_emitted / elapsed)
    return {
        "mb_per_s": mb_per_s,
        "tokens_per_s": tok_per_s,
        "elapsed_s": elapsed,
    }

