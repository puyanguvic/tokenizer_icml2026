#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from datasets import ClassLabel, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from run_ctok_experiment import resolve_keys


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    tokenizer_name = Path(args.tokenizer_path).resolve().name
    dataset_tag = args.dataset.replace("/", "__")
    return Path("results") / f"exp2_{dataset_tag}_{tokenizer_name}"


def resolve_label_mapping(train_ds, label_key: str) -> Tuple[List[str], Callable]:
    feature = train_ds.features.get(label_key)
    if isinstance(feature, ClassLabel):
        label_list = list(feature.names)

        def add_labels(batch):
            return {"labels": batch[label_key]}

        return label_list, add_labels

    labels = sorted(set(train_ds.unique(label_key)))
    label_list = [str(x) for x in labels]
    label_to_id = {name: i for i, name in enumerate(label_list)}

    def add_labels(batch):
        return {"labels": [label_to_id[str(x)] for x in batch[label_key]]}

    return label_list, add_labels


def run(args: argparse.Namespace) -> None:
    text_key, label_key = resolve_keys(args.dataset, args.text_key, args.label_key)
    if label_key is None:
        raise SystemExit("label_key is required for classification.")
    print(f"Using text_key='{text_key}' label_key='{label_key or ''}'")

    if args.config:
        train_ds = load_dataset(args.dataset, args.config, split=args.train_split, cache_dir=args.cache_dir)
    else:
        train_ds = load_dataset(args.dataset, split=args.train_split, cache_dir=args.cache_dir)

    try:
        if args.config:
            eval_ds = load_dataset(args.dataset, args.config, split=args.eval_split, cache_dir=args.cache_dir)
        else:
            eval_ds = load_dataset(args.dataset, split=args.eval_split, cache_dir=args.cache_dir)
    except Exception:
        print("Eval split not found; creating train/validation split.")
        split = train_ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]

    if args.sample_ratio < 1.0:
        train_keep = max(1, int(len(train_ds) * args.sample_ratio))
        eval_keep = max(1, int(len(eval_ds) * args.sample_ratio))
        print(f"Sampling train {train_keep}/{len(train_ds)} and eval {eval_keep}/{len(eval_ds)}")
        train_ds = train_ds.shuffle(seed=args.seed).select(range(train_keep))
        eval_ds = eval_ds.shuffle(seed=args.seed + 1).select(range(eval_keep))

    label_list, add_labels = resolve_label_mapping(train_ds, label_key)
    num_labels = len(label_list)
    print(f"Detected {num_labels} labels: {label_list}")

    train_ds = train_ds.map(add_labels, batched=True)
    eval_ds = eval_ds.map(add_labels, batched=True)
    if label_key != "labels":
        train_ds = train_ds.remove_columns([label_key])
        eval_ds = eval_ds.remove_columns([label_key])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    def preprocess(batch):
        return tokenizer(batch[text_key], truncation=True, max_length=args.max_length)

    remove_cols = [c for c in train_ds.column_names if c not in ["labels", text_key]]
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=remove_cols)
    remove_cols = [c for c in eval_ds.column_names if c not in ["labels", text_key]]
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=remove_cols)

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
        vocab_size=len(tokenizer),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = resolve_output_dir(args)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training output: {output_dir}")
    trainer.train()
    trainer.evaluate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_split", default="validation")
    ap.add_argument("--eval_ratio", type=float, default=0.1)
    ap.add_argument("--text_key", default="")
    ap.add_argument("--label_key", default="")
    ap.add_argument("--cache_dir", default="datasets/hf_cache")
    ap.add_argument("--sample_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--model_name", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=2)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
