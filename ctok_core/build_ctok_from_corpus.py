from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
import heapq
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import gc

from . import hygiene
from . import pretokenize

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

_HYGIENE_CFG: Optional[hygiene.HygieneConfig] = None
_PRETOK_CFG: Optional[pretokenize.PreTokenizerConfig] = None


def _init_hygiene_worker(cfg_dict: Dict[str, object]) -> None:
    global _HYGIENE_CFG
    _HYGIENE_CFG = hygiene.HygieneConfig.from_dict(cfg_dict)


def _apply_hygiene_sample(sample: Tuple[Optional[str], str]) -> Tuple[Optional[str], str]:
    y, x = sample
    if _HYGIENE_CFG is None:
        return y, x
    return y, hygiene.apply_hygiene(x, _HYGIENE_CFG)


def _init_pretok_worker(cfg_dict: Dict[str, object]) -> None:
    global _PRETOK_CFG
    _PRETOK_CFG = pretokenize.PreTokenizerConfig.from_dict(cfg_dict)


def _apply_pretok_sample(sample: Tuple[Optional[str], str]) -> Tuple[Optional[str], str]:
    y, x = sample
    if _PRETOK_CFG is None:
        return y, x
    return y, pretokenize.apply_pretokenize(x, _PRETOK_CFG)


def _collect_base_chars_chunk(args: Tuple[List[object], Set[str], int]) -> Set[str]:
    items, boundaries, max_base_chars = args
    chars: Set[str] = set(boundaries)
    for item in items:
        s = _get_text(item)
        for ch in s:
            chars.add(ch)
            if len(chars) >= max_base_chars:
                return chars
    return chars


def _progress(iterable, total=None, desc: str = "", unit: str = "it"):
    if _tqdm is None:
        return iterable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stdout,
        dynamic_ncols=True,
    )


def parse_boundaries(boundaries: str) -> Set[str]:
    # Interpret escapes (\n, \t, \" ...) using unicode_escape decoding.
    decoded = boundaries.encode("utf-8").decode("unicode_escape")
    return set(decoded)


def iter_txt(path: str, max_samples: Optional[int]) -> Iterable[Tuple[Optional[str], str]]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield None, line
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_tsv(path: str, max_samples: Optional[int], label_key: str = "label") -> Iterable[Tuple[str, str]]:
    # Format: label\ttext
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            y, x = parts[0], parts[1]
            yield y, x
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_jsonl(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    import json

    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            x = obj[text_key]
            y = obj[label_key] if label_key and label_key in obj else None
            yield (str(y) if y is not None else None), str(x)
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_parquet(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    # Prefer pyarrow.dataset for large datasets; fallback to pandas.
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        cols = [text_key] + ([label_key] if label_key else [])
        scanner = dataset.scanner(columns=cols)
        n = 0
        for batch in scanner.to_batches():
            table = batch.to_pydict()
            texts = table[text_key]
            labels = table[label_key] if label_key else [None] * len(texts)
            for y, x in zip(labels, texts):
                if x is None:
                    continue
                yield (str(y) if y is not None else None), str(x)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return
    except Exception:
        import pandas as pd

        df = pd.read_parquet(path, columns=[text_key] + ([label_key] if label_key else []))
        n = 0
        for row in df.itertuples(index=False):
            if label_key:
                y, x = getattr(row, label_key), getattr(row, text_key)
            else:
                x = getattr(row, text_key)
                y = None
            if x is None:
                continue
            yield (str(y) if y is not None else None), str(x)
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def corpus_iter(fmt: str, path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    if fmt == "txt":
        return iter_txt(path, max_samples)
    if fmt == "tsv":
        return iter_tsv(path, max_samples)
    if fmt == "jsonl":
        return iter_jsonl(path, max_samples, text_key=text_key, label_key=label_key)
    if fmt == "parquet":
        return iter_parquet(path, max_samples, text_key=text_key, label_key=label_key)
    raise ValueError(f"Unknown --format: {fmt}")


def _get_text(item: object) -> str:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[1])
    return str(item)


def _select_prefilter_indices(total: int, max_samples: int) -> List[int]:
    if max_samples <= 0 or max_samples >= total:
        return list(range(total))
    step = max(1, total // max_samples)
    idxs = list(range(0, total, step))
    return idxs[:max_samples]


def _build_prefixes(tokens: Iterable[str], boundaries: Set[str]) -> Set[str]:
    prefixes: Set[str] = set()
    for tok in tokens:
        if any(ch in boundaries for ch in tok):
            continue
        for i in range(1, len(tok) + 1):
            prefixes.add(tok[:i])
    return prefixes


def _get_memory_info_bytes() -> Tuple[Optional[int], Optional[int]]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    try:
        total = None
        available = None
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                if total is not None and available is not None:
                    break
        return total, available
    except Exception:
        return None, None


_GLOBAL_TEXTS: Optional[List[object]] = None
_GLOBAL_BOUNDARIES: Optional[Set[str]] = None
_GLOBAL_MAX_LEN: Optional[int] = None
_GLOBAL_ALLOW_BOUNDARY_AT_ENDS: Optional[bool] = None
_GLOBAL_MAX_CHARS_PER_SAMPLE: Optional[int] = None
_GLOBAL_CANDIDATE_WHITELIST: Optional[Set[str]] = None
_GLOBAL_CANDIDATE_PREFIXES: Optional[Set[str]] = None


def _init_candidates_globals(
    texts: List[object],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    whitelist: Optional[Set[str]] = None,
    prefixes: Optional[Set[str]] = None,
) -> None:
    global _GLOBAL_TEXTS, _GLOBAL_BOUNDARIES, _GLOBAL_MAX_LEN, _GLOBAL_ALLOW_BOUNDARY_AT_ENDS, _GLOBAL_MAX_CHARS_PER_SAMPLE
    global _GLOBAL_CANDIDATE_WHITELIST, _GLOBAL_CANDIDATE_PREFIXES
    _GLOBAL_TEXTS = texts
    _GLOBAL_BOUNDARIES = boundaries
    _GLOBAL_MAX_LEN = max_len
    _GLOBAL_ALLOW_BOUNDARY_AT_ENDS = allow_boundary_at_ends
    _GLOBAL_MAX_CHARS_PER_SAMPLE = max_chars_per_sample
    _GLOBAL_CANDIDATE_WHITELIST = whitelist
    _GLOBAL_CANDIDATE_PREFIXES = prefixes


def _collect_candidates_chunk_idxs(idxs: Iterable[int]) -> Counter[str]:
    texts = _GLOBAL_TEXTS
    boundaries = _GLOBAL_BOUNDARIES
    max_len = _GLOBAL_MAX_LEN
    allow_boundary_at_ends = _GLOBAL_ALLOW_BOUNDARY_AT_ENDS
    max_chars_per_sample = _GLOBAL_MAX_CHARS_PER_SAMPLE
    whitelist = _GLOBAL_CANDIDATE_WHITELIST
    prefixes = _GLOBAL_CANDIDATE_PREFIXES
    if texts is None or boundaries is None or max_len is None or allow_boundary_at_ends is None or max_chars_per_sample is None:
        raise RuntimeError("Candidate worker globals not initialized.")
    cnt: Counter[str] = Counter()
    for idx in idxs:
        text = _get_text(texts[idx])
        s = text[:max_chars_per_sample]
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            cur = ""
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                nxt = cur + s[j]
                if prefixes is not None and nxt not in prefixes:
                    break
                cur = nxt
                if len(cur) >= 2 and (whitelist is None or cur in whitelist):
                    cnt[cur] += 1
                j += 1
            if allow_boundary_at_ends and len(cur) >= 2:
                if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                    t = cur + s[j]
                    if whitelist is None or t in whitelist:
                        cnt[t] += 1
                if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                    t = s[i - 1] + cur
                    if whitelist is None or t in whitelist:
                        cnt[t] += 1
            i += 1
    return cnt


_GLOBAL_BASE_TEXTS: Optional[List[object]] = None
_GLOBAL_BASE_BOUNDARIES: Optional[Set[str]] = None
_GLOBAL_BASE_MAX_CHARS: Optional[int] = None


def _init_base_chars_globals(samples: List[object], boundaries: Set[str], max_base_chars: int) -> None:
    global _GLOBAL_BASE_TEXTS, _GLOBAL_BASE_BOUNDARIES, _GLOBAL_BASE_MAX_CHARS
    _GLOBAL_BASE_TEXTS = samples
    _GLOBAL_BASE_BOUNDARIES = boundaries
    _GLOBAL_BASE_MAX_CHARS = max_base_chars


def _collect_base_chars_chunk_idxs(idxs: Iterable[int]) -> Set[str]:
    samples = _GLOBAL_BASE_TEXTS
    boundaries = _GLOBAL_BASE_BOUNDARIES
    max_base_chars = _GLOBAL_BASE_MAX_CHARS
    if samples is None or boundaries is None or max_base_chars is None:
        raise RuntimeError("Base-char worker globals not initialized.")
    chars: Set[str] = set()
    for idx in idxs:
        s = _get_text(samples[idx])
        for ch in s:
            chars.add(ch)
            if len(chars) >= max_base_chars:
                return chars
    return chars


def collect_base_chars(
    samples: Iterable[object],
    boundaries: Set[str],
    max_base_chars: int,
    use_ascii_base: bool,
    extra_tokens: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 1,
    chunk_factor: int = 4,
) -> Set[str]:
    chars: Set[str] = set()
    if use_ascii_base:
        chars |= hygiene.ascii_base_chars()
    chars |= boundaries
    if extra_tokens:
        chars.update(extra_tokens)
    sample_list = samples if isinstance(samples, list) else list(samples)
    total = len(sample_list)
    if max_samples is not None:
        sample_list = sample_list[:max_samples]
        total = len(sample_list)

    if num_workers <= 1 or total == 0:
        iterator = _progress(sample_list, total=total, desc="Collecting base chars", unit="samples")
        for item in iterator:
            s = _get_text(item)
            for ch in s:
                chars.add(ch)
                if len(chars) >= max_base_chars:
                    return chars
        return chars

    print(f"Collecting base chars with {num_workers} workers")
    chunk_size = max(1, total // (num_workers * max(chunk_factor, 1)))
    num_chunks = (total + chunk_size - 1) // chunk_size
    idx_iter = (range(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size))
    ctx = mp.get_context()
    if ctx.get_start_method(allow_none=True) == "fork":
        _init_base_chars_globals(sample_list, boundaries, max_base_chars)
        pool = ctx.Pool(processes=num_workers)
    else:
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_base_chars_globals,
            initargs=(sample_list, boundaries, max_base_chars),
        )
    with pool:
        results_iter = pool.imap_unordered(_collect_base_chars_chunk_idxs, idx_iter, chunksize=1)
        if _tqdm is not None:
            results_iter = _tqdm(
                results_iter,
                total=num_chunks,
                desc="Collecting base chars (mp)",
                unit="chunks",
                file=sys.stdout,
                dynamic_ncols=True,
            )
        for cset in results_iter:
            chars.update(cset)
            if len(chars) >= max_base_chars:
                break
    return chars


def _collect_candidates_chunk(args: Tuple[List[object], Set[str], int, int, bool, int]) -> Counter[str]:
    items, boundaries, max_len, min_freq, allow_boundary_at_ends, max_chars_per_sample = args
    cnt: Counter[str] = Counter()
    for item in items:
        text = _get_text(item)
        s = text[:max_chars_per_sample]
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2:
                    cnt[cur] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        cnt[cur + s[j]] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        cnt[s[i - 1] + cur] += 1
            i += 1
    return cnt


def _chunk_by_char_budget(items: List[object], max_chars_per_sample: int, target_chars: int) -> List[List[object]]:
    chunks: List[List[object]] = []
    cur: List[object] = []
    cur_chars = 0
    for item in items:
        s = _get_text(item)
        cur.append(item)
        cur_chars += min(len(s), max_chars_per_sample)
        if cur_chars >= target_chars:
            chunks.append(cur)
            cur = []
            cur_chars = 0
    if cur:
        chunks.append(cur)
    return chunks


def _count_token_label_chunk(
    args: Tuple[List[int], List[Optional[str]], List[str], Set[str], int, bool, int, Set[str]]
) -> Tuple[Counter[str], Dict[str, Dict[str, int]]]:
    idxs, labels, texts, boundaries, max_len, allow_boundary_at_ends, max_chars_per_sample, top = args
    local_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    local_labels: Counter[str] = Counter()
    for i in idxs:
        y = labels[i]
        if y is None:
            continue
        x = texts[i]
        local_labels[y] += 1
        s = x[:max_chars_per_sample]
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2 and cur in top:
                    local_counts[cur][y] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if t in top:
                            local_counts[t][y] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if t in top:
                            local_counts[t][y] += 1
            i += 1
    return local_labels, {k: dict(v) for k, v in local_counts.items()}


def _collect_doc_stats_chunk(
    args: Tuple[List[object], Set[str], int, bool, int, Set[str]]
) -> Tuple[Counter[str], Dict[str, int]]:
    items, candidates, max_len, allow_boundary_at_ends, max_chars_per_sample, boundaries = args
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    for item in items:
        text = _get_text(item)
        s = text[:max_chars_per_sample]
        n = len(s)
        i = 0
        local: Counter[str] = Counter()
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2 and cur in candidates:
                    local[cur] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if t in candidates:
                            local[t] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if t in candidates:
                            local[t] += 1
            i += 1
        for tok in local:
            doc_freq[tok] += 1
            prev = max_in_doc.get(tok, 0)
            if local[tok] > prev:
                max_in_doc[tok] = local[tok]
    return doc_freq, max_in_doc


def collect_candidates(
    texts: Iterable[object],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    whitelist: Optional[Set[str]] = None,
    whitelist_prefixes: Optional[Set[str]] = None,
    num_workers: int = 1,
    mp_chunksize: int = 1,
    chunk_factor: int = 4,
    chunk_chars: int = 0,
) -> Counter[str]:
    cnt: Counter[str] = Counter()
    text_list = texts if isinstance(texts, list) else list(texts)
    total = len(text_list)
    if (whitelist is not None or whitelist_prefixes is not None) and chunk_chars:
        chunk_chars = 0
    if num_workers <= 1 or total == 0:
        for item in _progress(text_list, total=total, desc="Collecting candidates", unit="samples"):
            text = _get_text(item)
            s = text[:max_chars_per_sample]
            n = len(s)
            i = 0
            while i < n:
                if s[i] in boundaries:
                    i += 1
                    continue
                j = i
                cur = ""
                while j < n and (j - i) < max_len and s[j] not in boundaries:
                    nxt = cur + s[j]
                    if whitelist_prefixes is not None and nxt not in whitelist_prefixes:
                        break
                    cur = nxt
                    if len(cur) >= 2 and (whitelist is None or cur in whitelist):
                        cnt[cur] += 1
                    j += 1
                if allow_boundary_at_ends and len(cur) >= 2:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if whitelist is None or t in whitelist:
                            cnt[t] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if whitelist is None or t in whitelist:
                            cnt[t] += 1
                i += 1
    else:
        print(f"Collecting candidates with {num_workers} workers")
        if chunk_chars and chunk_chars > 0:
            chunks = _chunk_by_char_budget(text_list, max_chars_per_sample, chunk_chars)
            num_chunks = len(chunks)
            args_iter = (
                (c, boundaries, max_len, min_freq, allow_boundary_at_ends, max_chars_per_sample) for c in chunks
            )
            with mp.Pool(processes=num_workers) as pool:
                results_iter = pool.imap_unordered(_collect_candidates_chunk, args_iter, chunksize=mp_chunksize)
                if _tqdm is not None:
                    results_iter = _tqdm(
                        results_iter,
                        total=num_chunks,
                        desc="Collecting candidates (mp)",
                        unit="chunks",
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                for c in results_iter:
                    cnt.update(c)
            # filter
            for k in list(cnt.keys()):
                if cnt[k] < min_freq:
                    del cnt[k]
            return cnt
        else:
            chunk_size = max(1, total // (num_workers * max(chunk_factor, 1)))
            num_chunks = (total + chunk_size - 1) // chunk_size
            chunk_iter = (range(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size))
        ctx = mp.get_context()
        if ctx.get_start_method(allow_none=True) == "fork":
            _init_candidates_globals(
                text_list,
                boundaries,
                max_len,
                allow_boundary_at_ends,
                max_chars_per_sample,
                whitelist=whitelist,
                prefixes=whitelist_prefixes,
            )
            pool = ctx.Pool(processes=num_workers)
        else:
            pool = ctx.Pool(
                processes=num_workers,
                initializer=_init_candidates_globals,
                initargs=(
                    text_list,
                    boundaries,
                    max_len,
                    allow_boundary_at_ends,
                    max_chars_per_sample,
                    whitelist,
                    whitelist_prefixes,
                ),
            )
        with pool:
            results_iter = pool.imap_unordered(_collect_candidates_chunk_idxs, chunk_iter, chunksize=mp_chunksize)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=num_chunks,
                    desc="Collecting candidates (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            for c in results_iter:
                cnt.update(c)
    # filter
    for k in list(cnt.keys()):
        if cnt[k] < min_freq:
            del cnt[k]
    return cnt


def estimate_gain(token: str, freq: int) -> float:
    # With single-char base vocab, a token of length L saves (L-1) per occurrence.
    L = len(token)
    return float(freq) * max(L - 1, 0)


def compute_token_mi(
    token: str,
    label_counts: Dict[str, int],
    token_label_counts: Dict[str, Dict[str, int]],
) -> float:
    # MI approx: sum_y p(y|t) log p(y|t)/p(y)
    t_counts = token_label_counts.get(token)
    if not t_counts:
        return 0.0
    total_t = sum(t_counts.values())
    total = sum(label_counts.values())
    if total_t == 0 or total == 0:
        return 0.0
    mi = 0.0
    for y, c in t_counts.items():
        py_t = c / total_t
        py = label_counts.get(y, 0) / total
        if py_t > 0 and py > 0:
            mi += py_t * math.log(py_t / py)
    return mi


def build_vocab(
    base_chars: Set[str],
    candidates: Counter[str],
    vocab_size: int,
    special_tokens: Sequence[str],
    required_tokens: Sequence[str],
    semantic_mode: str,
    lambda_sem: float,
    label_counts: Dict[str, int],
    token_label_counts: Dict[str, Dict[str, int]],
    junk_penalty_beta: float = 0.0,
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    cur_id = 0

    # specials first
    for st in special_tokens:
        token_to_id[st] = cur_id
        cur_id += 1

    # required tokens (e.g., typed tokens)
    for tok in sorted(set(required_tokens)):
        if tok in token_to_id:
            continue
        token_to_id[tok] = cur_id
        cur_id += 1

    # base chars
    for ch in sorted(base_chars):
        if ch in token_to_id:
            continue
        token_to_id[ch] = cur_id
        cur_id += 1

    if len(token_to_id) >= vocab_size:
        # If base already exceeds budget, truncate deterministically.
        items = list(token_to_id.items())[:vocab_size]
        return {k: i for i, (k, _) in enumerate(items)}

    # Keep only top-K candidates to reduce peak memory.
    budget = max(0, vocab_size - len(token_to_id))
    if budget == 0:
        return token_to_id

    def _inv_utf8_bytes(s: str) -> bytes:
        b = s.encode("utf-8", errors="surrogatepass")
        return bytes((255 - x) for x in b)

    # Min-heap of worst elements (score asc, tok desc).
    heap: List[Tuple[float, bytes, str]] = []
    for tok, f in candidates.items():
        if tok in token_to_id:
            continue
        gain = estimate_gain(tok, f)
        if gain <= 0:
            continue
        score = gain
        if semantic_mode == "mi" and lambda_sem > 0 and label_counts:
            score += lambda_sem * compute_token_mi(tok, label_counts, token_label_counts)
        if junk_penalty_beta > 0:
            score -= junk_penalty_beta * hygiene.junk_score(tok)
        key = (score, _inv_utf8_bytes(tok), tok)
        if len(heap) < budget:
            heapq.heappush(heap, key)
        else:
            # Replace if strictly better (higher score; tie -> lexicographically smaller tok).
            if key > heap[0]:
                heapq.heapreplace(heap, key)

    # Deterministic finalize: sort by (-score, tok)
    heap.sort(key=lambda x: (-x[0], x[2]))
    for score, _inv, tok in heap:
        if len(token_to_id) >= vocab_size:
            break
        token_to_id[tok] = cur_id
        cur_id += 1

    return token_to_id


def collect_doc_stats(
    texts: Iterable[object],
    candidates: Set[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    num_workers: int = 1,
    mp_chunksize: int = 1,
    chunk_factor: int = 4,
) -> Tuple[Counter[str], Dict[str, int]]:
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    total = len(texts) if hasattr(texts, "__len__") else None
    if num_workers <= 1 or total == 0:
        for item in _progress(texts, total=total, desc="Doc stats", unit="samples"):
            text = _get_text(item)
            s = text[:max_chars_per_sample]
            n = len(s)
            i = 0
            local: Counter[str] = Counter()
            while i < n:
                if s[i] in boundaries:
                    i += 1
                    continue
                j = i
                while j < n and (j - i) < max_len and s[j] not in boundaries:
                    j += 1
                    cur = s[i:j]
                    if len(cur) >= 2 and cur in candidates:
                        local[cur] += 1
                    if allow_boundary_at_ends:
                        if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                            t = cur + s[j]
                            if t in candidates:
                                local[t] += 1
                        if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                            t = s[i - 1] + cur
                            if t in candidates:
                                local[t] += 1
                i += 1
            for tok in local:
                doc_freq[tok] += 1
                prev = max_in_doc.get(tok, 0)
                if local[tok] > prev:
                    max_in_doc[tok] = local[tok]
    else:
        print(f"Doc stats with {num_workers} workers")
        text_list = texts if isinstance(texts, list) else list(texts)
        chunk_size = max(1, len(text_list) // (num_workers * max(chunk_factor, 1)))
        chunks = [text_list[i : i + chunk_size] for i in range(0, len(text_list), chunk_size)]
        args_list = [(c, candidates, max_len, allow_boundary_at_ends, max_chars_per_sample, boundaries) for c in chunks]
        with mp.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_collect_doc_stats_chunk, args_list, chunksize=mp_chunksize)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Doc stats (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            for local_doc_freq, local_max_in_doc in results_iter:
                doc_freq.update(local_doc_freq)
                for tok, cnt in local_max_in_doc.items():
                    prev = max_in_doc.get(tok, 0)
                    if cnt > prev:
                        max_in_doc[tok] = cnt
    return doc_freq, max_in_doc


def filter_candidates(
    candidates: Counter[str],
    texts: Iterable[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    filter_value_fragments: bool,
    typed_tokens: Sequence[str],
    min_doc_freq: int,
    max_doc_concentration: float,
    num_workers: int = 1,
    mp_chunksize: int = 1,
    chunk_factor: int = 4,
) -> Counter[str]:
    filtered = Counter()
    items_iter = candidates.items()
    if _tqdm is not None:
        items_iter = _tqdm(
            items_iter,
            total=len(candidates),
            desc="Filtering candidates",
            unit="tokens",
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for tok, cnt in items_iter:
        if hygiene.is_typed_token_fragment(tok, typed_tokens):
            continue
        if filter_value_fragments and hygiene.is_value_fragment(tok):
            continue
        filtered[tok] = cnt

    if min_doc_freq <= 1 and max_doc_concentration >= 1.0:
        return filtered

    print("Computing candidate doc stats...")
    doc_freq, max_in_doc = collect_doc_stats(
        texts,
        candidates=set(filtered.keys()),
        boundaries=boundaries,
        max_len=max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=max_chars_per_sample,
        num_workers=num_workers,
        mp_chunksize=mp_chunksize,
        chunk_factor=chunk_factor,
    )
    out = Counter()
    out_iter = filtered.items()
    if _tqdm is not None:
        out_iter = _tqdm(
            out_iter,
            total=len(filtered),
            desc="Applying doc filters",
            unit="tokens",
            file=sys.stdout,
            dynamic_ncols=True,
        )
    for tok, cnt in out_iter:
        if min_doc_freq > 1 and doc_freq.get(tok, 0) < min_doc_freq:
            continue
        if max_doc_concentration < 1.0:
            ratio = max_in_doc.get(tok, 0) / max(cnt, 1)
            if ratio > max_doc_concentration:
                continue
        out[tok] = cnt
    return out


def build_token_label_counts(
    labels: List[Optional[str]],
    texts: List[str],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    semantic_top_k: int,
    candidates: Optional[Counter[str]] = None,
    num_workers: int = 1,
    mp_chunksize: int = 1,
    chunk_factor: int = 4,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    # Build candidate list first, then count token occurrences by label for top_k.
    usable_texts = [t for y, t in zip(labels, texts) if y is not None]
    if candidates is None:
        candidates = collect_candidates(
            texts=usable_texts,
            boundaries=boundaries,
            max_len=max_len,
            min_freq=min_freq,
            allow_boundary_at_ends=allow_boundary_at_ends,
            max_chars_per_sample=max_chars_per_sample,
            num_workers=num_workers,
            mp_chunksize=mp_chunksize,
        )
    top = set([tok for tok, _ in candidates.most_common(semantic_top_k)])

    label_counts: Dict[str, int] = Counter([y for y in labels if y is not None])
    token_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = len(texts) if hasattr(texts, "__len__") else None
    if num_workers <= 1 or total == 0:
        for i in _progress(range(total), total=total, desc="Counting token-labels", unit="samples"):
            y = labels[i]
            if y is None:
                continue
            x = texts[i]
            # simple presence counting in each sample: count all occurrences (not just binary) for cheap signal
            s = x[:max_chars_per_sample]
            n = len(s)
            i = 0
            while i < n:
                if s[i] in boundaries:
                    i += 1
                    continue
                j = i
                while j < n and (j - i) < max_len and s[j] not in boundaries:
                    j += 1
                    cur = s[i:j]
                    if len(cur) >= 2 and cur in top:
                        token_label_counts[cur][y] += 1
                    if allow_boundary_at_ends:
                        if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                            t = cur + s[j]
                            if t in top:
                                token_label_counts[t][y] += 1
                        if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                            t = s[i - 1] + cur
                            if t in top:
                                token_label_counts[t][y] += 1
                i += 1
    else:
        print(f"Counting token-labels with {num_workers} workers")
        chunk_size = max(1, total // (num_workers * max(chunk_factor, 1)))
        idxs = list(range(total))
        chunks = [idxs[i : i + chunk_size] for i in range(0, total, chunk_size)]
        args_list = [
            (c, labels, texts, boundaries, max_len, allow_boundary_at_ends, max_chars_per_sample, top)
            for c in chunks
        ]
        with mp.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_count_token_label_chunk, args_list, chunksize=mp_chunksize)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Counting token-labels (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            label_counts = Counter()
            token_label_counts = defaultdict(lambda: defaultdict(int))
            for local_labels, local_counts in results_iter:
                label_counts.update(local_labels)
                for tok, lab_map in local_counts.items():
                    for y, c in lab_map.items():
                        token_label_counts[tok][y] += c

    return dict(label_counts), {k: dict(v) for k, v in token_label_counts.items()}


def write_fast_tokenizer_json(outdir: str, token_to_id: Dict[str, int], special_tokens: Sequence[str]) -> None:
    # Build a tokenizers backend tokenizer.json using WordPiece greedy longest-match.
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.processors import TemplateProcessing

    unk = "[UNK]"
    if unk not in token_to_id:
        raise ValueError("[UNK] must be in vocab")

    model = WordPiece(vocab=token_to_id, unk_token=unk, continuing_subword_prefix="")
    tok = Tokenizer(model)

    # Add post-processor for CLS/SEP
    cls = "[CLS]"
    sep = "[SEP]"
    if cls in token_to_id and sep in token_to_id:
        tok.post_processor = TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[(cls, token_to_id[cls]), (sep, token_to_id[sep])],
        )

    tok.save(os.path.join(outdir, "tokenizer.json"))


def write_artifact(
    outdir: str,
    token_to_id: Dict[str, int],
    boundaries: Set[str],
    vocab_size: int,
    max_len: int,
    min_freq: int,
    fmt: str,
    text_key: str,
    label_key: Optional[str],
    semantic_mode: str,
    lambda_sem: float,
    semantic_top_k: int,
    model_max_length: int,
    emit_code: bool,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    hygiene_metrics: Dict[str, float],
    hygiene_build: Dict[str, object],
    build_perf: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # vocab.json for debugging / slow tokenizer
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=True, indent=2)

    if build_perf is None:
        build_perf = {}

    meta = {
        "match_special_tokens": False,
        "artifact_version": "ctok-fast-v1",
        "pipeline_locked": True,
        "lowercase": bool(hygiene_build.get("lowercase", False)),
        "hygiene": hygiene_cfg.to_dict(),
        "pretokenizer": pretok_cfg.to_dict(),
        "hygiene_metrics": hygiene_metrics,
        "build": {
            "format": fmt,
            "text_key": text_key,
            "label_key": label_key,
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": len(token_to_id),
            "max_len": max_len,
            "min_freq": min_freq,
            "boundaries": sorted(list(boundaries)),
            "semantic_mode": semantic_mode,
            "lambda_sem": lambda_sem,
            "semantic_top_k": semantic_top_k,
            "pretokenizer": "generic" if pretok_cfg.enabled else "none",
            "lowercase": bool(hygiene_build.get("lowercase", False)),
            **hygiene_build,
            **build_perf,
        },
    }
    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    # tokenizer_config.json: IMPORTANT auto_map tuple = [slow, fast]
    tok_cfg = {
        "tokenizer_class": "CTokTokenizerFast",
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_ctok.CTokTokenizer",
                "tokenization_ctok_fast.CTokTokenizerFast",
            ]
        },
        "model_max_length": model_max_length,
        "padding_side": "right",
        "truncation_side": "right",
    }
    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=True, indent=2)

    sp_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(outdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(sp_map, f, ensure_ascii=True, indent=2)

    # Fast tokenizer backend
    write_fast_tokenizer_json(outdir, token_to_id, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    if emit_code:
        here = Path(__file__).resolve().parent
        for fn in ["tokenization_ctok.py", "tokenization_ctok_fast.py", "hygiene.py", "pretokenize.py"]:
            shutil.copy(str(here / fn), os.path.join(outdir, fn))

    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# CTok Fast Tokenizer Artifact\n\n"
            "This directory is loadable via `AutoTokenizer.from_pretrained(path, trust_remote_code=True)`.\n\n"
            "Files:\n"
            "- tokenizer.json: Rust backend (WordPiece greedy longest-match with empty continuation prefix)\n"
            "- vocab.json: token->id (debug / slow tokenizer)\n"
            "- ctok_meta.json: build metadata\n"
            "- tokenizer_config.json, special_tokens_map.json: Transformers integration\n"
        )


def build_ctok_from_samples(
    samples: List[Tuple[Optional[str], str]],
    text_key: str,
    label_key: Optional[str],
    outdir: str,
    args: argparse.Namespace,
) -> None:
    boundaries = parse_boundaries(args.boundaries)
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)

    hygiene_cfg = hygiene.default_hygiene_config()
    hygiene_cfg.enabled = not args.no_hygiene
    if not hygiene_cfg.enabled:
        hygiene_cfg.typed_tokens = []
        hygiene_cfg.patterns = []

    pretok_cfg = pretokenize.default_pretokenizer_config()
    pretok_cfg.enabled = args.pretokenizer != "none"
    if not pretok_cfg.enabled:
        pretok_cfg.patterns = []

    def _materialize_texts_labels(
        iterable: Iterable[Tuple[Optional[str], str]],
        total: int,
        desc: str,
    ) -> Tuple[List[str], List[Optional[str]]]:
        texts: List[str] = []
        labels: List[Optional[str]] = []
        it = _progress(iterable, total=total, desc=desc, unit="samples")
        for y, x in it:
            labels.append(y)
            texts.append(x)
        return texts, labels

    if args.lowercase:
        samples = [(y, x.lower()) for y, x in samples]
    if hygiene_cfg.enabled:
        if num_workers > 1 and len(samples) > 0:
            print(f"Applying hygiene with {num_workers} workers")
            cfg_dict = hygiene_cfg.to_dict()
            with mp.Pool(processes=num_workers, initializer=_init_hygiene_worker, initargs=(cfg_dict,)) as pool:
                results_iter = pool.imap(_apply_hygiene_sample, samples, chunksize=args.mp_chunksize)
                if _tqdm is not None:
                    results_iter = _tqdm(
                        results_iter,
                        total=len(samples),
                        desc="Applying hygiene",
                        unit="samples",
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                samples = list(results_iter)
        else:
            samples = [(y, hygiene.apply_hygiene(x, hygiene_cfg)) for y, x in samples]
    if pretok_cfg.enabled:
        if num_workers > 1 and len(samples) > 0:
            print(f"Applying pretokenizer with {num_workers} workers")
            cfg_dict = pretok_cfg.to_dict()
            with mp.Pool(processes=num_workers, initializer=_init_pretok_worker, initargs=(cfg_dict,)) as pool:
                results_iter = pool.imap(_apply_pretok_sample, samples, chunksize=args.mp_chunksize)
                if _tqdm is not None:
                    results_iter = _tqdm(
                        results_iter,
                        total=len(samples),
                        desc="Applying pretokenizer",
                        unit="samples",
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                texts, labels = _materialize_texts_labels(results_iter, len(samples), "Collecting pretokenized")
        else:
            texts, labels = _materialize_texts_labels(
                ((y, pretokenize.apply_pretokenize(x, pretok_cfg)) for y, x in samples),
                len(samples),
                "Applying pretokenizer",
            )
        print("Releasing pretokenizer samples...")
        del samples
        gc.collect()
        print("Proceeding to base char collection...")
    else:
        texts, labels = _materialize_texts_labels(samples, len(samples), "Preparing text/labels")
        print("Releasing samples...")
        del samples
        gc.collect()
        print("Proceeding to base char collection...")

    base_chars = collect_base_chars(
        texts,
        boundaries,
        max_base_chars=args.max_base_chars,
        use_ascii_base=args.use_ascii_base,
        extra_tokens=hygiene_cfg.typed_tokens,
        max_samples=args.base_chars_max_samples,
        num_workers=num_workers,
        chunk_factor=args.mp_chunk_factor,
    )

    allow_boundary_at_ends = not args.no_boundary_ends

    candidate_whitelist: Optional[Set[str]] = None
    candidate_prefixes: Optional[Set[str]] = None
    prefilter_samples = args.candidate_prefilter_samples
    prefilter_min_freq = args.candidate_prefilter_min_freq
    total_texts = len(texts)
    if prefilter_samples <= 0:
        _total_mem, avail_mem = _get_memory_info_bytes()
        if total_texts >= 1_000_000 and args.max_len >= 8 and args.max_chars_per_sample >= 1024:
            avail_gb = (avail_mem or 0) / (1024**3)
            if avail_mem is None or avail_gb < 24:
                if avail_mem is None:
                    prefilter_samples = min(500_000, total_texts)
                elif avail_gb < 8:
                    prefilter_samples = min(200_000, total_texts)
                elif avail_gb < 16:
                    prefilter_samples = min(400_000, total_texts)
                else:
                    prefilter_samples = min(800_000, total_texts)
                if prefilter_min_freq <= 0:
                    prefilter_min_freq = 2 if args.min_freq >= 5 else 1
                mem_note = "unknown" if avail_mem is None else f"{avail_gb:.1f}GB"
                print(
                    f"Auto prefiltering candidates with {prefilter_samples} samples "
                    f"(min_freq={prefilter_min_freq}, avail_mem={mem_note})"
                )
    if prefilter_samples and prefilter_samples > 0 and prefilter_samples < total_texts:
        idxs = _select_prefilter_indices(total_texts, prefilter_samples)
        sample_texts = [texts[i] for i in idxs]
        print(f"Prefiltering candidates with {len(sample_texts)} samples")
        pre_cands = collect_candidates(
            texts=sample_texts,
            boundaries=boundaries,
            max_len=args.max_len,
            min_freq=args.min_freq,
            allow_boundary_at_ends=allow_boundary_at_ends,
            max_chars_per_sample=args.max_chars_per_sample,
            num_workers=num_workers,
            mp_chunksize=args.mp_chunksize,
            chunk_factor=args.mp_chunk_factor,
            chunk_chars=args.mp_chunk_chars,
        )
        pre_min = max(1, int(prefilter_min_freq))
        if pre_min > 1:
            for k in list(pre_cands.keys()):
                if pre_cands[k] < pre_min:
                    del pre_cands[k]
        candidate_whitelist = set(pre_cands.keys())
        candidate_prefixes = _build_prefixes(candidate_whitelist, boundaries)
        print(f"Prefilter candidates kept: {len(candidate_whitelist)}")

    cands = collect_candidates(
        texts=texts,
        boundaries=boundaries,
        max_len=args.max_len,
        min_freq=args.min_freq,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=args.max_chars_per_sample,
        whitelist=candidate_whitelist,
        whitelist_prefixes=candidate_prefixes,
        num_workers=num_workers,
        mp_chunksize=args.mp_chunksize,
        chunk_factor=args.mp_chunk_factor,
        chunk_chars=args.mp_chunk_chars,
    )
    cands_raw = cands
    cands = filter_candidates(
        candidates=cands,
        texts=texts,
        boundaries=boundaries,
        max_len=args.max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=args.max_chars_per_sample,
        filter_value_fragments=not args.no_filter_value_fragments,
        typed_tokens=hygiene_cfg.typed_tokens,
        min_doc_freq=args.min_doc_freq,
        max_doc_concentration=args.max_doc_concentration,
        num_workers=num_workers,
        mp_chunksize=args.mp_chunksize,
        chunk_factor=args.mp_chunk_factor,
    )

    label_counts: Dict[str, int] = {}
    token_label_counts: Dict[str, Dict[str, int]] = {}

    if args.semantic_mode == "mi" and label_key is not None:
        label_counts, token_label_counts = build_token_label_counts(
            labels=labels,
            texts=texts,
            boundaries=boundaries,
            max_len=args.max_len,
            min_freq=args.min_freq,
            allow_boundary_at_ends=allow_boundary_at_ends,
            max_chars_per_sample=args.max_chars_per_sample,
            semantic_top_k=args.semantic_top_k,
            candidates=cands_raw,
            num_workers=num_workers,
            mp_chunksize=args.mp_chunksize,
            chunk_factor=args.mp_chunk_factor,
        )
        if cands:
            token_label_counts = {k: v for k, v in token_label_counts.items() if k in cands}

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    vocab = build_vocab(
        base_chars=base_chars,
        candidates=cands,
        vocab_size=args.vocab_size,
        special_tokens=special,
        required_tokens=hygiene_cfg.typed_tokens,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        label_counts=label_counts,
        token_label_counts=token_label_counts,
        junk_penalty_beta=args.junk_penalty_beta,
    )

    write_artifact(
        outdir=outdir,
        token_to_id=vocab,
        boundaries=boundaries,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        fmt=args.format,
        text_key=text_key,
        label_key=label_key,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        semantic_top_k=args.semantic_top_k,
        model_max_length=args.model_max_length,
        emit_code=args.emit_code,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        hygiene_metrics=hygiene.vocab_hygiene_metrics(vocab.keys(), hygiene_cfg.typed_tokens),
        hygiene_build={
            "hygiene_enabled": hygiene_cfg.enabled,
            "filter_value_fragments": not args.no_filter_value_fragments,
            "min_doc_freq": args.min_doc_freq,
            "max_doc_concentration": args.max_doc_concentration,
            "junk_penalty_beta": args.junk_penalty_beta,
            "lowercase": bool(args.lowercase),
        },
    )

    print(f"Wrote CTok FAST artifact to: {outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(cands)}")


# =========================
# Locked, user-friendly builder (C-class optimizations)
# =========================

_PERF_POLICY = "locked-v1"
_CACHE_NAME = "_ctok_preprocessed.jsonl.gz"
_CACHE_META = "_ctok_preprocessed.meta.json"

# Hard-coded performance defaults (auto tuned).
_MAX_WORKERS = 16
_CHUNK_CHARS = 1_500_000  # ~1.5MB of text per task
_BASE_CHARS_MAX_SAMPLES = 200_000
_HEAVY_HITTER_MIN_K = 200_000
_HEAVY_HITTER_MAX_K = 800_000
_HEAVY_HITTER_FACTOR = 20


def _auto_workers(user_override: int = 0) -> int:
    if user_override and user_override > 0:
        return max(1, min(_MAX_WORKERS, int(user_override)))
    n = os.cpu_count() or 4
    n = max(1, n - 1)
    return max(1, min(_MAX_WORKERS, n))


def _sha1_json(obj: Dict[str, object]) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def _preprocess_text(x: str, *, lowercase: bool, hygiene_cfg: hygiene.HygieneConfig, pretok_cfg: pretokenize.PreTokenizerConfig) -> str:
    if lowercase:
        x = x.lower()
    if hygiene_cfg.enabled:
        x = hygiene.apply_hygiene(x, hygiene_cfg)
    if pretok_cfg.enabled:
        x = pretokenize.apply_pretokenize(x, pretok_cfg)
    return x


def _ensure_preprocessed_cache(
    *,
    args: argparse.Namespace,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    boundaries: Set[str],
    label_key: Optional[str],
) -> Tuple[str, int, Set[str]]:
    """Create or reuse a preprocessed cache.

    The cache stores already (lowercase+hygiene+pretok) processed text truncated to max_chars_per_sample,
    so multi-pass builds do not redo expensive regex work.
    """
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, _CACHE_NAME)
    meta_path = os.path.join(outdir, _CACHE_META)

    cfg = {
        "policy": _PERF_POLICY,
        "corpus": os.path.abspath(args.corpus),
        "format": args.format,
        "text_key": args.text_key,
        "label_key": label_key,
        "lowercase": bool(args.lowercase),
        "hygiene_enabled": bool(hygiene_cfg.enabled),
        "hygiene_version": getattr(hygiene_cfg, "version", ""),
        "pretok_enabled": bool(pretok_cfg.enabled),
        "pretok_version": getattr(pretok_cfg, "version", ""),
        "max_chars_per_sample": int(args.max_chars_per_sample),
        "boundaries": args.boundaries,
        "use_ascii_base": bool(args.use_ascii_base),
        "max_base_chars": int(args.max_base_chars),
        "max_samples": int(args.max_samples or 0),
    }
    cfg_hash = _sha1_json(cfg)

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("cfg_hash") == cfg_hash:
                base_chars = set(meta.get("base_chars", []))
                total = int(meta.get("total_samples", 0))
                if total > 0 and base_chars:
                    return cache_path, total, base_chars
        except Exception:
            pass

    # Build cache
    print(f"[ctok] Building preprocessed cache: {cache_path}")
    it = corpus_iter(args.format, args.corpus, args.max_samples, args.text_key, label_key)

    base_chars: Set[str] = set(boundaries)
    if args.use_ascii_base:
        base_chars.update(chr(i) for i in range(128))
    total = 0
    sampled = 0

    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        for y, x in _progress(it, desc="Preprocessing", unit="samples"):
            if not x:
                continue
            x = _preprocess_text(
                str(x),
                lowercase=bool(args.lowercase),
                hygiene_cfg=hygiene_cfg,
                pretok_cfg=pretok_cfg,
            )
            if args.max_chars_per_sample and args.max_chars_per_sample > 0:
                x = x[: int(args.max_chars_per_sample)]
            if not x:
                continue
            rec = {"x": x, "y": y}
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            total += 1

            if sampled < _BASE_CHARS_MAX_SAMPLES and len(base_chars) < int(args.max_base_chars):
                sampled += 1
                for ch in x:
                    base_chars.add(ch)
                    if len(base_chars) >= int(args.max_base_chars):
                        break

    meta = {
        "cfg_hash": cfg_hash,
        "cfg": cfg,
        "total_samples": total,
        "base_chars": sorted(list(base_chars)),
        "created_at": time.time(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    return cache_path, total, base_chars


def _iter_cache(cache_path: str) -> Iterable[Tuple[Optional[str], str]]:
    with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj.get("y"), obj.get("x", "")


def _iter_candidate_tokens(s: str, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool) -> Iterable[str]:
    n = len(s)
    i = 0
    while i < n:
        if s[i] in boundaries:
            i += 1
            continue
        j = i
        cur = ""
        while j < n and (j - i) < max_len and s[j] not in boundaries:
            cur = cur + s[j]
            if len(cur) >= 2:
                yield cur
            j += 1
        if allow_boundary_at_ends and len(cur) >= 2:
            if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                yield cur + s[j]
            if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                yield s[i - 1] + cur
        i += 1


class SpaceSaving:
    """SpaceSaving heavy-hitter sketch.

    Keeps at most k items; counts are upper bounds with bounded error.
    """

    def __init__(self, k: int) -> None:
        self.k = max(1, int(k))
        self.counts: Dict[str, int] = {}

    def offer(self, item: str, w: int = 1) -> None:
        if w <= 0:
            return
        c = self.counts.get(item)
        if c is not None:
            self.counts[item] = c + w
            return
        if len(self.counts) < self.k:
            self.counts[item] = w
            return
        # Replace current minimum.
        m_item = min(self.counts, key=self.counts.get)
        m_cnt = self.counts[m_item]
        del self.counts[m_item]
        self.counts[item] = m_cnt + w


class _TrieNode:
    __slots__ = ("children", "core", "count_core", "pre", "suf")

    def __init__(self) -> None:
        self.children: Dict[str, _TrieNode] = {}
        self.core: Optional[str] = None
        self.count_core: bool = False
        self.pre: Optional[Set[str]] = None  # boundary chars that can prefix this core
        self.suf: Optional[Set[str]] = None  # boundary chars that can suffix this core


class TokenTrie:
    def __init__(self) -> None:
        self.root = _TrieNode()

    def _insert_core(self, core: str) -> _TrieNode:
        node = self.root
        for ch in core:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
        node.core = core
        return node

    @classmethod
    def from_full_tokens(cls, tokens: Iterable[str], boundaries: Set[str]) -> "TokenTrie":
        trie = cls()
        for tok in tokens:
            if not tok or len(tok) < 2:
                continue
            # prefix/suffix boundary handling
            if tok[0] in boundaries and len(tok) >= 3:
                b = tok[0]
                core = tok[1:]
                node = trie._insert_core(core)
                node.pre = node.pre or set()
                node.pre.add(b)
            elif tok[-1] in boundaries and len(tok) >= 3:
                b = tok[-1]
                core = tok[:-1]
                node = trie._insert_core(core)
                node.suf = node.suf or set()
                node.suf.add(b)
            else:
                core = tok
                node = trie._insert_core(core)
                node.count_core = True
        return trie


def _chunk_cache_texts(cache_path: str, *, char_budget: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    chars = 0
    for _y, x in _iter_cache(cache_path):
        if not x:
            continue
        chunk.append(x)
        chars += len(x)
        if chars >= char_budget and chunk:
            yield chunk
            chunk = []
            chars = 0
    if chunk:
        yield chunk


def _chunk_cache_samples(cache_path: str, *, char_budget: int) -> Iterable[List[Tuple[Optional[str], str]]]:
    chunk: List[Tuple[Optional[str], str]] = []
    chars = 0
    for y, x in _iter_cache(cache_path):
        if not x:
            continue
        chunk.append((y, x))
        chars += len(x)
        if chars >= char_budget and chunk:
            yield chunk
            chunk = []
            chars = 0
    if chunk:
        yield chunk


def _scan_texts_counts(texts: List[str], trie: TokenTrie, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool) -> Counter[str]:
    out: Counter[str] = Counter()
    for s in texts:
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            node = trie.root
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                nxt = node.children.get(s[j])
                if nxt is None:
                    break
                node = nxt
                j += 1
                if node.core is not None:
                    core = node.core
                    if node.count_core:
                        out[core] += 1
                    if allow_boundary_at_ends:
                        if node.suf is not None and j < n and s[j] in node.suf:
                            out[core + s[j]] += 1
                        if node.pre is not None and i > 0 and s[i - 1] in node.pre:
                            out[s[i - 1] + core] += 1
            i += 1
    return out


_MP_TRIE: Optional[TokenTrie] = None
_MP_BOUNDARIES: Optional[Set[str]] = None
_MP_MAX_LEN: int = 0
_MP_ALLOW_BOUNDS: bool = False


def _init_mp_scan(trie: TokenTrie, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool) -> None:
    global _MP_TRIE, _MP_BOUNDARIES, _MP_MAX_LEN, _MP_ALLOW_BOUNDS
    _MP_TRIE = trie
    _MP_BOUNDARIES = boundaries
    _MP_MAX_LEN = int(max_len)
    _MP_ALLOW_BOUNDS = bool(allow_boundary_at_ends)


def _clear_mp_scan() -> None:
    global _MP_TRIE, _MP_BOUNDARIES
    _MP_TRIE = None
    _MP_BOUNDARIES = None


def _mp_count_chunk(texts: List[str]) -> Counter[str]:
    assert _MP_TRIE is not None and _MP_BOUNDARIES is not None
    return _scan_texts_counts(texts, _MP_TRIE, _MP_BOUNDARIES, _MP_MAX_LEN, _MP_ALLOW_BOUNDS)


def _mp_doc_stats_chunk(texts: List[str]) -> Tuple[Counter[str], Dict[str, int]]:
    assert _MP_TRIE is not None and _MP_BOUNDARIES is not None
    return _scan_texts_doc_stats(texts, _MP_TRIE, _MP_BOUNDARIES, _MP_MAX_LEN, _MP_ALLOW_BOUNDS)


def _mp_token_label_chunk(samples: List[Tuple[Optional[str], str]]) -> Tuple[Counter[str], Dict[str, Dict[str, int]]]:
    assert _MP_TRIE is not None and _MP_BOUNDARIES is not None
    local_labels: Counter[str] = Counter()
    local_tlc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for y, x in samples:
        if y is None:
            continue
        local_labels[y] += 1
        c = _scan_texts_counts([x], _MP_TRIE, _MP_BOUNDARIES, _MP_MAX_LEN, _MP_ALLOW_BOUNDS)
        for tok, v in c.items():
            dst = local_tlc[tok]
            dst[y] = dst.get(y, 0) + v
    return local_labels, {k: dict(v) for k, v in local_tlc.items()}


def _count_candidates_stream(cache_path: str, trie: TokenTrie, *, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool, workers: int) -> Counter[str]:
    # Use multiprocessing only on fork systems to avoid pickling large tries.
    start = mp.get_start_method(allow_none=True)
    use_mp = workers > 1 and start == "fork"
    cnt: Counter[str] = Counter()

    chunk_iter = _chunk_cache_texts(cache_path, char_budget=_CHUNK_CHARS)
    if not use_mp:
        for chunk in _progress(chunk_iter, desc="Counting candidates", unit="chunks"):
            cnt.update(_scan_texts_counts(chunk, trie, boundaries, max_len, allow_boundary_at_ends))
        return cnt

    print(f"[ctok] Counting candidates with {workers} workers ({start})")
    _init_mp_scan(trie, boundaries, max_len, allow_boundary_at_ends)
    try:
        with mp.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_mp_count_chunk, chunk_iter, chunksize=1)
            if _tqdm is not None:
                results = _tqdm(results, desc="Counting candidates (mp)", unit="chunks", file=sys.stdout, dynamic_ncols=True)
            for c in results:
                cnt.update(c)
        return cnt
    finally:
        _clear_mp_scan()


def _scan_texts_doc_stats(texts: List[str], trie: TokenTrie, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool) -> Tuple[Counter[str], Dict[str, int]]:
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    for s in texts:
        local = _scan_texts_counts([s], trie, boundaries, max_len, allow_boundary_at_ends)
        for tok, c in local.items():
            doc_freq[tok] += 1
            prev = max_in_doc.get(tok, 0)
            if c > prev:
                max_in_doc[tok] = c
    return doc_freq, max_in_doc


def _scan_samples_token_labels(
    samples: List[Tuple[Optional[str], str]],
    trie: TokenTrie,
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
) -> Tuple[Counter[str], Dict[str, Dict[str, int]]]:
    local_labels: Counter[str] = Counter()
    local_tlc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for y, x in samples:
        if y is None:
            continue
        local_labels[y] += 1
        c = _scan_texts_counts([x], trie, boundaries, max_len, allow_boundary_at_ends)
        for tok, v in c.items():
            dst = local_tlc[tok]
            dst[y] = dst.get(y, 0) + v
    return local_labels, {k: dict(v) for k, v in local_tlc.items()}


def _collect_doc_stats_stream(cache_path: str, trie: TokenTrie, *, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool, workers: int) -> Tuple[Counter[str], Dict[str, int]]:
    start = mp.get_start_method(allow_none=True)
    use_mp = workers > 1 and start == "fork"

    chunk_iter = _chunk_cache_texts(cache_path, char_budget=_CHUNK_CHARS)
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}

    if not use_mp:
        for chunk in _progress(chunk_iter, desc="Doc stats", unit="chunks"):
            df, mid = _scan_texts_doc_stats(chunk, trie, boundaries, max_len, allow_boundary_at_ends)
            doc_freq.update(df)
            for tok, c in mid.items():
                prev = max_in_doc.get(tok, 0)
                if c > prev:
                    max_in_doc[tok] = c
        return doc_freq, max_in_doc

    print(f"[ctok] Doc stats with {workers} workers ({start})")
    _init_mp_scan(trie, boundaries, max_len, allow_boundary_at_ends)
    try:
        with mp.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_mp_doc_stats_chunk, chunk_iter, chunksize=1)
            if _tqdm is not None:
                results = _tqdm(results, desc="Doc stats (mp)", unit="chunks", file=sys.stdout, dynamic_ncols=True)
            for df, mid in results:
                doc_freq.update(df)
                for tok, c in mid.items():
                    prev = max_in_doc.get(tok, 0)
                    if c > prev:
                        max_in_doc[tok] = c
        return doc_freq, max_in_doc
    finally:
        _clear_mp_scan()


def _build_token_label_counts_stream(cache_path: str, trie: TokenTrie, *, boundaries: Set[str], max_len: int, allow_boundary_at_ends: bool, workers: int) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    start = mp.get_start_method(allow_none=True)
    use_mp = workers > 1 and start == "fork"

    label_counts: Counter[str] = Counter()
    token_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    chunk_iter = _chunk_cache_samples(cache_path, char_budget=_CHUNK_CHARS)

    if not use_mp:
        for chunk in _progress(chunk_iter, desc="Token-label counts", unit="chunks"):
            lc, tlc = _scan_samples_token_labels(chunk, trie, boundaries, max_len, allow_boundary_at_ends)
            label_counts.update(lc)
            for tok, d in tlc.items():
                dst = token_label_counts[tok]
                for y, v in d.items():
                    dst[y] = dst.get(y, 0) + v
        return dict(label_counts), {k: dict(v) for k, v in token_label_counts.items()}

    print(f"[ctok] Token-label counts with {workers} workers ({start})")
    _init_mp_scan(trie, boundaries, max_len, allow_boundary_at_ends)
    try:
        with mp.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_mp_token_label_chunk, chunk_iter, chunksize=1)
            if _tqdm is not None:
                results = _tqdm(results, desc="Token-label counts (mp)", unit="chunks", file=sys.stdout, dynamic_ncols=True)
            for lc, tlc in results:
                label_counts.update(lc)
                for tok, d in tlc.items():
                    dst = token_label_counts[tok]
                    for y, v in d.items():
                        dst[y] = dst.get(y, 0) + v
        return dict(label_counts), {k: dict(v) for k, v in token_label_counts.items()}
    finally:
        _clear_mp_scan()


def build_ctok_from_corpus_locked(args: argparse.Namespace) -> None:
    """Locked builder: auto-tunes performance and keeps the CLI simple."""
    boundaries = parse_boundaries(args.boundaries)
    allow_boundary_at_ends = not bool(args.no_boundary_ends)

    label_key = args.label_key if args.label_key else None

    hygiene_cfg = hygiene.default_hygiene_config()
    hygiene_cfg.enabled = not bool(args.no_hygiene)
    if not hygiene_cfg.enabled:
        hygiene_cfg.typed_tokens = []
        hygiene_cfg.patterns = []

    pretok_cfg = pretokenize.default_pretokenizer_config()
    pretok_cfg.enabled = args.pretokenizer != "none"
    if not pretok_cfg.enabled:
        pretok_cfg.patterns = []

    workers = _auto_workers(int(getattr(args, "num_workers", 0) or 0))

    cache_path, total_samples, base_chars = _ensure_preprocessed_cache(
        args=args,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        boundaries=boundaries,
        label_key=label_key,
    )

    # Heavy-hitter candidate selection (C-class): bounded memory.
    approx_k = max(_HEAVY_HITTER_MIN_K, int(args.vocab_size) * _HEAVY_HITTER_FACTOR, int(args.semantic_top_k) * 2)
    approx_k = min(_HEAVY_HITTER_MAX_K, approx_k)
    print(f"[ctok] Heavy-hitter pass (k={approx_k})")
    ss = SpaceSaving(approx_k)
    for _y, x in _progress(_iter_cache(cache_path), total=total_samples or None, desc="Heavy hitters", unit="samples"):
        if not x:
            continue
        for tok in _iter_candidate_tokens(x, boundaries, int(args.max_len), allow_boundary_at_ends):
            ss.offer(tok, 1)
    approx_tokens = set(ss.counts.keys())
    print(f"[ctok] Approx candidate set: {len(approx_tokens)}")

    # Exact counting only on approx set.
    trie = TokenTrie.from_full_tokens(approx_tokens, boundaries)
    cands = _count_candidates_stream(
        cache_path,
        trie,
        boundaries=boundaries,
        max_len=int(args.max_len),
        allow_boundary_at_ends=allow_boundary_at_ends,
        workers=workers,
    )
    # Apply min_freq early.
    for k in list(cands.keys()):
        if cands[k] < int(args.min_freq):
            del cands[k]

    # Hygiene-based candidate pruning.
    filtered = Counter()
    for tok, cnt in cands.items():
        if hygiene.is_typed_token_fragment(tok, hygiene_cfg.typed_tokens):
            continue
        if (not args.no_filter_value_fragments) and hygiene.is_value_fragment(tok):
            continue
        filtered[tok] = cnt
    cands = filtered

    # Doc filters (streaming) if enabled.
    if int(args.min_doc_freq) > 1 or float(args.max_doc_concentration) < 1.0:
        print("[ctok] Computing doc stats for candidate filtering...")
        trie_df = TokenTrie.from_full_tokens(set(cands.keys()), boundaries)
        doc_freq, max_in_doc = _collect_doc_stats_stream(
            cache_path,
            trie_df,
            boundaries=boundaries,
            max_len=int(args.max_len),
            allow_boundary_at_ends=allow_boundary_at_ends,
            workers=workers,
        )
        out = Counter()
        for tok, cnt in _progress(cands.items(), total=len(cands), desc="Applying doc filters", unit="tokens"):
            if int(args.min_doc_freq) > 1 and doc_freq.get(tok, 0) < int(args.min_doc_freq):
                continue
            if float(args.max_doc_concentration) < 1.0:
                ratio = max_in_doc.get(tok, 0) / max(cnt, 1)
                if ratio > float(args.max_doc_concentration):
                    continue
            out[tok] = cnt
        cands = out

    # MI counts if enabled and labels are available.
    label_counts: Dict[str, int] = {}
    token_label_counts: Dict[str, Dict[str, int]] = {}
    if args.semantic_mode == "mi" and float(args.lambda_sem) > 0 and label_key is not None:
        top_tokens = [tok for tok, _ in cands.most_common(int(args.semantic_top_k))]
        if top_tokens:
            print(f"[ctok] Token-label counting on top-{len(top_tokens)}")
            trie_top = TokenTrie.from_full_tokens(set(top_tokens), boundaries)
            label_counts, token_label_counts = _build_token_label_counts_stream(
                cache_path,
                trie_top,
                boundaries=boundaries,
                max_len=int(args.max_len),
                allow_boundary_at_ends=allow_boundary_at_ends,
                workers=workers,
            )

    # Base chars: ensure boundaries + optional ASCII, clamp.
    if args.use_ascii_base:
        base_chars.update(chr(i) for i in range(128))
    base_chars.update(boundaries)
    if len(base_chars) > int(args.max_base_chars):
        base_chars = set(sorted(base_chars)[: int(args.max_base_chars)])

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = build_vocab(
        base_chars=base_chars,
        candidates=cands,
        vocab_size=int(args.vocab_size),
        special_tokens=special,
        required_tokens=hygiene_cfg.typed_tokens,
        semantic_mode=args.semantic_mode,
        lambda_sem=float(args.lambda_sem),
        label_counts=label_counts,
        token_label_counts=token_label_counts,
        junk_penalty_beta=float(args.junk_penalty_beta),
    )

    build_perf = {
        "perf_policy": _PERF_POLICY,
        "workers": workers,
        "chunk_chars": _CHUNK_CHARS,
        "hh_k": approx_k,
        "cache": os.path.basename(cache_path),
        "passes": "cache+hh+count+doc(+mi)",
    }

    write_artifact(
        outdir=args.outdir,
        token_to_id=vocab,
        boundaries=boundaries,
        vocab_size=int(args.vocab_size),
        max_len=int(args.max_len),
        min_freq=int(args.min_freq),
        fmt=args.format,
        text_key=args.text_key,
        label_key=label_key,
        semantic_mode=args.semantic_mode,
        lambda_sem=float(args.lambda_sem),
        semantic_top_k=int(args.semantic_top_k),
        model_max_length=int(args.model_max_length),
        emit_code=bool(args.emit_code),
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        hygiene_metrics=hygiene.vocab_hygiene_metrics(vocab.keys(), hygiene_cfg.typed_tokens),
        hygiene_build={
            "hygiene_enabled": hygiene_cfg.enabled,
            "filter_value_fragments": not args.no_filter_value_fragments,
            "min_doc_freq": int(args.min_doc_freq),
            "max_doc_concentration": float(args.max_doc_concentration),
            "junk_penalty_beta": float(args.junk_penalty_beta),
            "lowercase": bool(args.lowercase),
        },
        build_perf=build_perf,
    )

    print(f"Wrote CTok FAST artifact to: {args.outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(cands)}")


def main() -> None:
    # User-facing CLI is intentionally simple: performance knobs are auto-tuned and locked.
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus (txt/tsv/jsonl/parquet) or parquet directory")
    ap.add_argument("--format", default="parquet", choices=["txt", "tsv", "jsonl", "parquet"])
    ap.add_argument("--text_key", default="text", help="For jsonl/parquet: text field")
    ap.add_argument("--label_key", default="label", help="For jsonl/parquet: label field; set to empty to disable")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_freq", type=int, default=50)
    ap.add_argument("--max_chars_per_sample", type=int, default=4096)

    ap.add_argument("--boundaries", type=str, default="=&?:/\\n\\t <>\\\"'", help="Boundary characters (supports escapes)")
    ap.add_argument("--no_boundary_ends", action="store_true")

    ap.add_argument("--use_ascii_base", action="store_true", help="Include ASCII chars (0..127) in base vocab")
    ap.add_argument("--max_base_chars", type=int, default=4096)

    ap.add_argument("--semantic_mode", choices=["none", "mi"], default="none")
    ap.add_argument("--lambda_sem", type=float, default=0.0)
    ap.add_argument("--semantic_top_k", type=int, default=50000)

    ap.add_argument("--no_hygiene", action="store_true", help="Disable hygiene replacements")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before hygiene/pretokenization")
    ap.add_argument("--pretokenizer", choices=["none", "generic"], default="none")

    ap.add_argument("--no_filter_value_fragments", action="store_true", help="Disable value-fragment candidate filtering")
    ap.add_argument("--min_doc_freq", type=int, default=1)
    ap.add_argument("--max_doc_concentration", type=float, default=1.0)
    ap.add_argument("--junk_penalty_beta", type=float, default=0.5)

    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--emit_code", action="store_true")

    # Hidden/dev-only flags (kept for compatibility, but not documented).
    ap.add_argument("--max_samples", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num_workers", type=int, default=0, help=argparse.SUPPRESS)

    args = ap.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    build_ctok_from_corpus_locked(args)


if __name__ == "__main__":
    main()
