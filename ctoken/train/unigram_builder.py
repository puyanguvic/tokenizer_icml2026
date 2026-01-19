from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    corpus_parquet: str
    outdir: str
    builder_py: Optional[str] = None

    vocab_size: int = 2048
    model_max_length: int = 512

    max_len: int = 16
    sample_words: int = 200000
    prune_iters: int = 4
    prune_frac: float = 0.20
    num_workers: int = 0

    lowercase: bool = False
    text_key: str = "text"
    max_samples: Optional[int] = None
    seed: int = 0


def _build_cmd(args: TrainArgs) -> list[str]:
    cmd = [
        "--format",
        "parquet",
        "--corpus",
        args.corpus_parquet,
        "--text_key",
        args.text_key,
        "--outdir",
        args.outdir,
        "--vocab_size",
        str(args.vocab_size),
        "--model_max_length",
        str(args.model_max_length),
        "--max_len",
        str(args.max_len),
        "--sample_words",
        str(args.sample_words),
        "--prune_iters",
        str(args.prune_iters),
        "--prune_frac",
        str(args.prune_frac),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.lowercase:
        cmd.append("--lowercase")
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    return cmd


def run_unigram_builder(args: TrainArgs) -> None:
    if args.builder_py:
        cmd = ["python", args.builder_py, *_build_cmd(args)]
        subprocess.check_call(cmd)
        return

    from ctoken.train import build_unigram_artifact

    ret = build_unigram_artifact.main(_build_cmd(args))
    if ret != 0:
        raise RuntimeError(f"Internal builder failed with exit code {ret}")
