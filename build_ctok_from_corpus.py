from __future__ import annotations

import argparse
import codecs
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


def math_log(x: float) -> float:
    # tiny wrapper (kept as a function for easy monkeypatching/testing)
    return math.log(x)


def to_bytestr(text: str) -> str:
    """unicode -> UTF-8 bytes -> latin-1 str (1 char == 1 byte)."""
    return text.encode("utf-8").decode("latin-1")


def decode_escapes(s: str) -> str:
    """Decode common escape sequences (e.g., \n and \t) using unicode_escape."""
    return codecs.decode(s, "unicode_escape")


def parse_boundaries(boundaries: str) -> Set[str]:
    decoded = decode_escapes(boundaries)
    return set(decoded)


def iter_records(
    corpus_path: str,
    fmt: str,
    text_key: str,
    label_key: str,
    max_samples: Optional[int],
) -> Iterator[Tuple[Optional[str], str]]:
    """Yield (label, text) records.

    fmt:
      - txt: each line is a sample; label=None
      - jsonl: each line is a JSON object; text=obj[text_key], label=obj.get(label_key)
      - tsv: each line: <label>\t<text>
      - parquet: a parquet file or directory; text=row[text_key], label=row.get(label_key)

    Labels are returned as strings (caller can map to integers if needed).
    """
    if fmt == "parquet":
        # Streaming parquet reader (preferred): pyarrow.dataset
        # Fallback: pandas.read_parquet
        n = 0
        try:
            import pyarrow.dataset as ds  # type: ignore
        except Exception:
            ds = None  # type: ignore

        if ds is not None:
            dataset = ds.dataset(corpus_path, format="parquet")
            cols = [text_key]
            if label_key:
                cols.append(label_key)
            # Scan in record batches to avoid loading everything into memory.
            scanner = dataset.scan(columns=cols, batch_size=8192)
            for batch in scanner.to_batches():
                # Convert columns to python lists.
                texts = batch.column(text_key).to_pylist()
                labels = None
                if label_key and label_key in batch.schema.names:
                    labels = batch.column(label_key).to_pylist()
                for i, text_val in enumerate(texts):
                    if text_val is None:
                        continue
                    text = str(text_val)
                    label: Optional[str] = None
                    if labels is not None:
                        lab = labels[i]
                        if lab is not None:
                            label = str(lab)
                    yield label, text
                    n += 1
                    if max_samples is not None and n >= max_samples:
                        return
            return

        # pandas fallback (loads selected columns into memory)
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Parquet support requires either 'pyarrow' (recommended) or 'pandas'. "
                "Install pyarrow for streaming reads."
            ) from e

        cols = [text_key]
        if label_key:
            cols.append(label_key)
        df = pd.read_parquet(corpus_path, columns=cols)
        for _, row in df.iterrows():
            text_val = row.get(text_key)
            if text_val is None:
                continue
            text = str(text_val)
            label_val = row.get(label_key) if label_key else None
            label = None if label_val is None else str(label_val)
            yield label, text
            n += 1
            if max_samples is not None and n >= max_samples:
                return

        return

    n = 0
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            label: Optional[str]
            text: str

            if fmt == "txt":
                label, text = None, line
            elif fmt == "jsonl":
                obj = json.loads(line)
                if text_key not in obj:
                    continue
                text = str(obj[text_key])
                label = None if label_key not in obj else str(obj[label_key])
            elif fmt == "tsv":
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                label, text = parts[0], parts[1]
            else:
                raise ValueError(f"Unknown format: {fmt}")

            yield label, text
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def collect_candidates(
    texts: Iterable[str],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_bytes_per_sample: int,
) -> Counter[str]:
    """Count boundary-aware substring candidates (byte-strings in latin-1).

    Rule:
      - internal characters of candidate must NOT include boundary chars
      - optionally allow one boundary char at the left or right end (e.g., "src=")

    Notes:
      - This is a fast prototype that works well for structured ASCII-heavy corpora.
      - For huge corpora, you can subsample via --max_samples.
    """
    cnt: Counter[str] = Counter()

    for text in texts:
        s = to_bytestr(text)
        if max_bytes_per_sample > 0:
            s = s[:max_bytes_per_sample]
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
                    # right boundary attached
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        cnt[cur + s[j]] += 1
                    # left boundary attached
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        cnt[s[i - 1] + cur] += 1

            i += 1

    # filter
    for k in list(cnt.keys()):
        if cnt[k] < min_freq:
            del cnt[k]

    return cnt


def _candidate_presence_per_label(
    records: Iterable[Tuple[Optional[str], str]],
    candidate_set: Set[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_bytes_per_sample: int,
) -> Tuple[Counter[str], Dict[str, Counter[str]]]:
    """Second-pass: compute sample-level presence counts for candidates.

    Returns:
      - present_total[c] = #samples where c appears at least once
      - present_by_label[label][c] = #samples with that label where c appears

    We enumerate boundary-aware substrings (same as candidate generation) and
    record which candidates appear in each sample. This is O(total_bytes * max_len)
    and does NOT iterate over candidates.
    """
    present_total: Counter[str] = Counter()
    present_by_label: Dict[str, Counter[str]] = {}

    for label, text in records:
        s = to_bytestr(text)
        if max_bytes_per_sample > 0:
            s = s[:max_bytes_per_sample]
        n = len(s)

        seen: Set[str] = set()
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue

            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2 and cur in candidate_set:
                    seen.add(cur)

                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        tok = cur + s[j]
                        if tok in candidate_set:
                            seen.add(tok)
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        tok = s[i - 1] + cur
                        if tok in candidate_set:
                            seen.add(tok)

            i += 1

        if not seen:
            continue

        for c in seen:
            present_total[c] += 1
            if label is not None:
                if label not in present_by_label:
                    present_by_label[label] = Counter()
                present_by_label[label][c] += 1

    return present_total, present_by_label


def estimate_label_mi(
    records: List[Tuple[str, str]],
    candidates: Counter[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_bytes_per_sample: int,
    top_k: int = 50000,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Estimate MI(label; presence(token)) for a subset of candidates.

    This is a lightweight semantic proxy you can use as a stand-in for
    probe-based distortion control (delta(c)). High MI indicates the token is
    informative about the label, so adding it tends to *reduce* semantic loss.

    We compute MI on binary feature F_c = 1[token c appears in sample].

    Args:
      records: list of (label, text)
      candidates: global frequency counts
      top_k: compute MI only for top_k candidates by frequency (keeps build fast)
      alpha: additive smoothing for probabilities
    """
    # Pick candidate subset by frequency
    most_common = candidates.most_common(top_k)
    cand_set: Set[str] = {tok for tok, _ in most_common}

    # Label counts
    label_counts: Counter[str] = Counter([lab for lab, _ in records])
    labels = sorted(label_counts.keys())
    n = len(records)
    if n == 0 or len(labels) <= 1:
        return {tok: 0.0 for tok in cand_set}

    present_total, present_by_label = _candidate_presence_per_label(
        records=records,
        candidate_set=cand_set,
        boundaries=boundaries,
        max_len=max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_bytes_per_sample=max_bytes_per_sample,
    )

    # Precompute denominators for smoothed probabilities
    L = len(labels)
    # P(l)
    p_l: Dict[str, float] = {}
    for lab in labels:
        p_l[lab] = (label_counts[lab] + alpha) / (n + alpha * L)

    mi: Dict[str, float] = {}
    for tok in cand_set:
        n1 = present_total.get(tok, 0)
        n0 = n - n1
        # P(f)
        p_f1 = (n1 + alpha) / (n + 2 * alpha)
        p_f0 = (n0 + alpha) / (n + 2 * alpha)

        val = 0.0
        for lab in labels:
            n_l = label_counts[lab]
            n_l1 = present_by_label.get(lab, Counter()).get(tok, 0)
            n_l0 = n_l - n_l1

            # P(l,f)
            p_lf1 = (n_l1 + alpha) / (n + alpha * L * 2)
            p_lf0 = (n_l0 + alpha) / (n + alpha * L * 2)

            # MI terms: sum_{l,f} p(l,f) log p(l,f)/(p(l)p(f))
            # Guard against tiny numerical issues; probabilities are positive under smoothing.
            val += p_lf1 * (0.0 if p_lf1 <= 0 else (math_log(p_lf1) - math_log(p_l[lab]) - math_log(p_f1)))
            val += p_lf0 * (0.0 if p_lf0 <= 0 else (math_log(p_lf0) - math_log(p_l[lab]) - math_log(p_f0)))

        mi[tok] = float(val)

    return mi


def estimate_gain(token: str, freq: int) -> int:
    """Compression-first surrogate gain vs byte-level base vocab.

    If base tokens are single bytes, a token of length L saves (L-1) per occurrence.
    This ignores overlap/shadowing; it's a decent fast baseline.
    """
    L = len(token)
    return freq * max(L - 1, 0)


def build_vocab(
    candidates: Counter[str],
    vocab_size: int,
    special_tokens: List[str],
    include_all_bytes: bool,
    semantic_mi: Optional[Dict[str, float]] = None,
    lambda_sem: float = 0.0,
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    cur = 0

    # specials first
    for st in special_tokens:
        token_to_id[st] = cur
        cur += 1

    # full 256 byte base vocab (recommended for complete coverage)
    if include_all_bytes:
        for b in range(256):
            tok = chr(b)
            if tok not in token_to_id:
                token_to_id[tok] = cur
                cur += 1

    base_size = len(token_to_id)
    if vocab_size <= base_size:
        raise ValueError(f"vocab_size={vocab_size} too small; base_size={base_size}")

    scored: List[Tuple[float, str]] = []
    for tok, f in candidates.items():
        if tok in token_to_id:
            continue
        gain = float(estimate_gain(tok, f))
        if gain <= 0:
            continue
        mi = 0.0
        if semantic_mi is not None and lambda_sem != 0.0:
            mi = float(semantic_mi.get(tok, 0.0))
        # Score = compression gain + lambda * MI(label; presence(token))
        # MI is a lightweight proxy for "semantic predictability"; higher MI is better.
        score = gain + lambda_sem * mi
        scored.append((score, tok))

    scored.sort(reverse=True)

    for score, tok in scored:
        if len(token_to_id) >= vocab_size:
            break
        token_to_id[tok] = cur
        cur += 1

    return token_to_id


def write_artifact(
    outdir: str,
    token_to_id: Dict[str, int],
    boundaries: Set[str],
    vocab_size: int,
    max_len: int,
    min_freq: int,
    model_max_length: int,
    copy_code_from: Optional[str],
    semantic_info: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # vocab
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=True, indent=2)

    # meta
    boundary_bytes = sorted({ord(ch) for ch in boundaries})
    meta: Dict[str, Any] = {
        "use_bytestr": True,
        "tiebreak": "min_token_id",
        "match_special_tokens": False,
        "artifact_version": "ctok-v1",
        "build": {
            "vocab_size": vocab_size,
            "max_len": max_len,
            "min_freq": min_freq,
            "boundary_bytes": boundary_bytes,
        },
    }
    if semantic_info is not None:
        meta["semantic"] = semantic_info
    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    # tokenizer config for AutoTokenizer (requires trust_remote_code=True)
    tok_cfg = {
        "tokenizer_class": "CTokTokenizer",
        "auto_map": {"AutoTokenizer": "tokenization_ctok.CTokTokenizer"},
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

    # copy tokenizer code for out-of-the-box loading
    if copy_code_from is not None:
        dst = os.path.join(outdir, "tokenization_ctok.py")
        with open(copy_code_from, "r", encoding="utf-8") as src_f, open(dst, "w", encoding="utf-8") as dst_f:
            dst_f.write(src_f.read())

    # README
    title = "# CTok Tokenizer Artifact\n\n"
    notes = (
        "This directory is directly loadable by Transformers.\n\n"
        "## Load\n"
        "```python\n"
        "from transformers import AutoTokenizer\n"
        "tok = AutoTokenizer.from_pretrained('./THIS_DIR', trust_remote_code=True)\n"
        "```\n\n"
        "## Files\n"
        "- vocab.json: token->id (includes specials + 256 bytes + induced tokens)\n"
        "- ctok_meta.json: build metadata (boundaries etc.)\n"
        "- tokenizer_config.json + special_tokens_map.json: loading config\n"
        "- tokenization_ctok.py: runtime implementation\n\n"
    )

    if semantic_info is None or semantic_info.get("mode", "none") == "none":
        extra = (
            "## Build objective\n"
            "This artifact was built with a compression-first surrogate (lambda_sem=0).\n"
            "\n"
            "To match the paper more closely, enable label-aware semantic scoring or plug in a probe.\n"
        )
    else:
        extra = (
            "## Build objective\n"
            "This artifact was built with label-aware semantic scoring (MI proxy).\n"
            f"- mode: {semantic_info.get('mode')}\n"
            f"- lambda_sem: {semantic_info.get('lambda_sem')}\n"
            f"- top_k: {semantic_info.get('top_k')}\n"
            "\n"
            "MI(label; token_presence) is a lightweight stand-in for probe-based semantic distortion control.\n"
        )

    readme = title + notes + extra
    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Corpus file path")
    ap.add_argument("--format", choices=["txt", "jsonl", "tsv", "parquet"], default="txt")
    ap.add_argument(
        "--text_key",
        default="text",
        help="For jsonl/parquet: which key/column holds text",
    )
    ap.add_argument(
        "--label_key",
        default="label",
        help="For jsonl/parquet: which key/column holds label (optional)",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12, help="Max candidate length (bytes)")
    ap.add_argument("--min_freq", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=200000)
    ap.add_argument(
        "--boundaries",
        type=str,
        default="=&?:/\\n\\t <>\\\"'<>",
        help="Boundary characters (supports escapes like \\n, \\t, \\\")",
    )
    ap.add_argument("--no_boundary_ends", action="store_true", help="Disallow boundary-at-ends candidates")
    ap.add_argument("--max_bytes_per_sample", type=int, default=4096)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--no_full_byte_base", action="store_true", help="Disable full 256-byte base vocab")
    ap.add_argument(
        "--semantic_mode",
        choices=["none", "mi"],
        default="none",
        help="Optional label-aware semantic scoring: 'mi' uses MI(label; token_presence) as a proxy",
    )
    ap.add_argument("--lambda_sem", type=float, default=0.0, help="Weight on semantic score (MI proxy)")
    ap.add_argument("--semantic_top_k", type=int, default=50000, help="Compute semantic scores for top-K frequent candidates")
    ap.add_argument("--semantic_alpha", type=float, default=1.0, help="Additive smoothing for MI estimation")
    ap.add_argument(
        "--emit_code",
        action="store_true",
        help="Copy tokenization_ctok.py into outdir for immediate loading",
    )
    args = ap.parse_args()

    boundaries = parse_boundaries(args.boundaries)

    # Load records. If semantic_mode != none, we materialize to a list for the second pass.
    rec_iter = iter_records(
        corpus_path=args.corpus,
        fmt=args.format,
        text_key=args.text_key,
        label_key=args.label_key,
        max_samples=args.max_samples,
    )

    records_list: Optional[List[Tuple[str, str]]] = None
    if args.semantic_mode != "none" and args.lambda_sem != 0.0:
        tmp: List[Tuple[str, str]] = []
        for lab, txt in rec_iter:
            if lab is None:
                continue
            tmp.append((lab, txt))
        records_list = tmp
        texts_for_cands = (t for _, t in records_list)
    else:
        texts_for_cands = (t for _, t in rec_iter)

    cands = collect_candidates(
        texts=texts_for_cands,
        boundaries=boundaries,
        max_len=args.max_len,
        min_freq=args.min_freq,
        allow_boundary_at_ends=not args.no_boundary_ends,
        max_bytes_per_sample=args.max_bytes_per_sample,
    )

    semantic_mi: Optional[Dict[str, float]] = None
    semantic_info: Optional[Dict[str, Any]] = None
    if args.semantic_mode == "mi" and args.lambda_sem != 0.0:
        if records_list is None or len(records_list) == 0:
            print("[WARN] semantic_mode=mi but no labeled records were found; falling back to compression-only.")
        else:
            semantic_mi = estimate_label_mi(
                records=records_list,
                candidates=cands,
                boundaries=boundaries,
                max_len=args.max_len,
                allow_boundary_at_ends=not args.no_boundary_ends,
                max_bytes_per_sample=args.max_bytes_per_sample,
                top_k=args.semantic_top_k,
                alpha=args.semantic_alpha,
            )
            semantic_info = {
                "mode": "mi",
                "lambda_sem": args.lambda_sem,
                "top_k": args.semantic_top_k,
                "alpha": args.semantic_alpha,
                "label_source": f"{args.format}:{args.label_key}" if args.format == "jsonl" else args.format,
            }

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = build_vocab(
        candidates=cands,
        vocab_size=args.vocab_size,
        special_tokens=special,
        include_all_bytes=not args.no_full_byte_base,
        semantic_mi=semantic_mi,
        lambda_sem=args.lambda_sem,
    )

    copy_code = None
    if args.emit_code:
        # tokenization_ctok.py assumed to be next to this script
        here = os.path.dirname(os.path.abspath(__file__))
        copy_code = os.path.join(here, "tokenization_ctok.py")

    write_artifact(
        outdir=args.outdir,
        token_to_id=vocab,
        boundaries=boundaries,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        model_max_length=args.model_max_length,
        copy_code_from=copy_code,
        semantic_info=semantic_info,
    )

    print(f"Wrote CTok artifact to: {args.outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(cands)}")


if __name__ == "__main__":
    main()
