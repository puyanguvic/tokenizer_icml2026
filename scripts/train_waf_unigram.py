#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a WAF Unigram tokenizer from waf_data_v2.")
    ap.add_argument("--split", default="train", help="Dataset split to use.")
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of rows.")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--max_piece_length", type=int, default=16)
    ap.add_argument("--no_progress", action="store_true", help="Disable trainer progress output.")
    ap.add_argument("--verify", action="store_true", help="Load the tokenizer via AutoTokenizer after training.")
    return ap.parse_args()


def row_to_text(row: Dict[str, object]) -> str:
    def get(key: str) -> str:
        val = row.get(key)
        return "" if val is None else str(val)

    return (
        f"<METHOD> {get('method')}\n"
        f"<URL> {get('url')}\n"
        f"<PROT> {get('protocol')}\n"
        f"<HDR>\n{get('headers')}\n"
        f"<BODY>\n{get('body')}\n"
    )


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers import normalizers, pre_tokenizers, processors
    from tokenizers.models import Unigram
    from tokenizers.trainers import UnigramTrainer
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / "datasets" / "hf_cache"
    outdir = repo_root / "tokenizers" / "waf_unigram"

    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading dataset waf_data_v2 ({args.split})")
    ds = load_dataset("puyang2025/waf_data_v2", split=args.split, cache_dir=str(cache_dir))
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    length = len(ds)

    def iterator():
        for row in ds:
            yield row_to_text(row)

    print(f"[2/3] Training Unigram tokenizer -> {outdir}")
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = UnigramTrainer(
        vocab_size=args.vocab_size,
        show_progress=not args.no_progress,
        special_tokens=SPECIAL_TOKENS,
        unk_token="[UNK]",
        max_piece_length=args.max_piece_length,
    )

    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=length)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer.save(str(outdir / "tokenizer.json"))

    with open(outdir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_max_length": args.model_max_length,
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            f,
            indent=2,
        )

    with open(outdir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            f,
            indent=2,
        )

    print(f"[3/3] Saved tokenizer to {outdir}")

    if args.verify:
        tok = AutoTokenizer.from_pretrained(str(outdir))
        print("Loaded tokenizer type:", type(tok))
        print("backend_tokenizer type:", type(tok.backend_tokenizer))


if __name__ == "__main__":
    main()
