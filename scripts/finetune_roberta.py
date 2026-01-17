#!/usr/bin/env python3
"""Fine-tune a RoBERTa model with a ctok tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ctok_extras.datasets.config import DatasetConfig
from ctok_extras.models.train import train_roberta


def _load_dataset_config(path: Path) -> DatasetConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a mapping.")
    return DatasetConfig(**payload)


def _load_model_name(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model config must be a mapping.")
    hf_id = payload.get("hf_id")
    if not hf_id:
        raise ValueError("Model config must include 'hf_id'.")
    return str(hf_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa with ctok.")
    parser.add_argument("--dataset-config", required=True, help="Dataset config YAML.")
    parser.add_argument("--model-config", required=True, help="Model config YAML.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer artifact dir.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-split", default="train", help="Train split name.")
    parser.add_argument("--eval-split", default="validation", help="Eval split name.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples.")
    args = parser.parse_args()

    dataset_config = _load_dataset_config(Path(args.dataset_config))
    model_name = _load_model_name(Path(args.model_config))

    metrics = train_roberta(
        dataset_config=dataset_config,
        model_name=model_name,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
