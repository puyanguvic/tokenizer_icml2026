from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from ctoken.io.parquet import ParquetSource, iter_record_batches
from ctoken.preprocess.registry import available as available_pre, get as get_pre
from ctoken.train.unigram_builder import TrainArgs, run_unigram_builder


def write_aligned_parquet(src_parquet: str, dst_parquet: str, preprocess: str, batch_size: int = 8192) -> None:
    spec, fn = get_pre(preprocess)

    src = ParquetSource(path=src_parquet, columns=spec.required_columns, batch_size=batch_size)

    writer = None
    try:
        for batch in iter_record_batches(src):
            out_batch = fn(batch)
            table = out_batch.to_table()
            if writer is None:
                Path(dst_parquet).parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(dst_parquet, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    # Ensure preprocessors register.
    import ctoken.preprocess  # noqa: F401

    p = argparse.ArgumentParser("ctoken parquet-only trainer")

    p.add_argument("--src_parquet", required=True, help="Input parquet (raw schema)")
    p.add_argument(
        "--preprocess",
        required=True,
        help=f"Preprocess name. Available: {', '.join(available_pre())}",
    )
    p.add_argument("--work_parquet", required=True, help="Output aligned parquet with a 'text' column")
    p.add_argument(
        "--builder_py",
        default=None,
        help="Optional path to an external build_unigram_artifact.py (defaults to ctoken's internal builder)",
    )
    p.add_argument("--outdir", required=True, help="Tokenizer artifact output directory")

    p.add_argument("--vocab_size", type=int, default=2048)
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--max_len", type=int, default=16)
    p.add_argument("--sample_words", type=int, default=200000)
    p.add_argument("--prune_iters", type=int, default=4)
    p.add_argument("--prune_frac", type=float, default=0.20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    print("[1/2] Aligning parquet ->", args.work_parquet)
    write_aligned_parquet(args.src_parquet, args.work_parquet, args.preprocess, batch_size=args.batch_size)

    print("[2/2] Training tokenizer artifact ->", args.outdir)
    run_unigram_builder(
        TrainArgs(
            corpus_parquet=args.work_parquet,
            outdir=args.outdir,
            builder_py=args.builder_py,
            vocab_size=args.vocab_size,
            model_max_length=args.model_max_length,
            max_len=args.max_len,
            sample_words=args.sample_words,
            prune_iters=args.prune_iters,
            prune_frac=args.prune_frac,
            num_workers=args.num_workers,
            lowercase=args.lowercase,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
