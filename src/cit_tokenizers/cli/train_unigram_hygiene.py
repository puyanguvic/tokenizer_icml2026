from __future__ import annotations
import argparse
from .common import add_common_corpus_args, add_common_contract_args, add_hygiene_artifact_args, add_logging_args, setup_logging_from_args, load_contract_config
from ..baselines.unigram_hygiene.trainer import train_unigram_hygiene

def main():
    ap = argparse.ArgumentParser()
    add_common_corpus_args(ap)
    add_common_contract_args(ap)
    add_hygiene_artifact_args(ap)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--model-max-length", type=int, default=512)
    args = ap.parse_args()

    cfg = load_contract_config(args)
    train_unigram_hygiene(
        corpus=args.corpus,
        outdir=args.outdir,
        vocab_size=args.vocab_size,
        contract_cfg=cfg,
        fmt=args.format,
        text_key=args.text_key,
        max_samples=args.max_samples,
        min_frequency=args.min_frequency,
        model_max_length=args.model_max_length,
        hygiene_outdir=args.hygiene_outdir,
        tokenizer_version=args.tokenizer_version,
        hygiene_version=args.hygiene_version,
        version=args.version,
        emit_contract_in_tokenizer_dir=bool(args.emit_contract),
    )

if __name__ == "__main__":
    main()
