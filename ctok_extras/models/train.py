"""Model training helpers."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any

from ctok_extras.datasets.config import DatasetConfig
from ctok_extras.datasets.io import load_dataset as load_local_dataset
from ctok_extras.models.encoders import load_roberta_classifier


def train_roberta(
    dataset_config: DatasetConfig,
    model_name: str,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    max_length: int = 512,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    train_split: str = "train",
    eval_split: str = "validation",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    datasets = _require_datasets()
    transformers = _require_transformers()
    from ctok_core.tokenization.hf import CtokHFTokenizer

    tokenizer = CtokHFTokenizer.from_pretrained(tokenizer_path)

    train_cfg = replace(dataset_config, hf_split=train_split, max_samples=max_train_samples)
    eval_cfg = replace(dataset_config, hf_split=eval_split, max_samples=max_eval_samples)

    train_dataset = _load_hf_dataset(train_cfg, datasets)
    eval_dataset = _load_hf_dataset(eval_cfg, datasets)

    label2id, id2label = _build_label_mapping(
        train_dataset,
        dataset_config.label_field,
        datasets,
        dataset_config.label_mapping,
    )
    train_dataset = _attach_labels(train_dataset, dataset_config.label_field, label2id)
    eval_dataset = _attach_labels(eval_dataset, dataset_config.label_field, label2id)

    train_dataset = train_dataset.map(
        lambda batch: _tokenize_batch(batch, dataset_config, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: _tokenize_batch(batch, dataset_config, tokenizer, max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    model = load_roberta_classifier(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "save_strategy": "epoch",
        "logging_steps": 100,
        "report_to": [],
    }
    args_params = inspect.signature(transformers.TrainingArguments.__init__).parameters
    if "evaluation_strategy" in args_params:
        args_kwargs["evaluation_strategy"] = "epoch"
    else:
        args_kwargs["eval_strategy"] = "epoch"

    args = transformers.TrainingArguments(**args_kwargs)

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics


def _require_datasets() -> Any:
    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise ImportError("Install 'datasets' to run training.") from exc
    return datasets


def _require_transformers() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("Install 'torch' to run training.")
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise ImportError("Install 'transformers' to run training.") from exc
    return transformers


def _load_hf_dataset(config: DatasetConfig, datasets: Any) -> Any:
    if config.hf_dataset:
        if config.hf_streaming and config.max_samples is None:
            raise ValueError("max_samples is required for streaming datasets.")
        dataset = datasets.load_dataset(
            config.hf_dataset,
            name=config.hf_name,
            split=config.hf_split,
            revision=config.hf_revision,
            streaming=config.hf_streaming,
        )
        if config.max_samples is not None:
            if config.hf_streaming:
                dataset = datasets.Dataset.from_list(list(dataset.take(config.max_samples)))
            else:
                dataset = dataset.select(range(config.max_samples))
        return dataset

    if not config.path:
        raise ValueError("Dataset path is required for non-HF datasets.")

    examples = load_local_dataset(config)
    return datasets.Dataset.from_list(
        [
            {
                config.text_field: ex.text,
                config.label_field: ex.label,
            }
            for ex in examples
        ]
    )


def _build_label_mapping(
    dataset: Any,
    label_field: str | None,
    datasets: Any,
    label_mapping: dict[str, int] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    if label_field is None:
        raise ValueError("label_field must be set for supervised training.")
    if label_mapping:
        label2id = {str(label): int(idx) for label, idx in label_mapping.items()}
        if len(set(label2id.values())) != len(label2id):
            raise ValueError("label_mapping has duplicate ids.")
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label
    features = getattr(dataset, "features", {})
    if label_field in features and isinstance(features[label_field], datasets.ClassLabel):
        names = list(features[label_field].names)
        label2id = {str(idx): idx for idx in range(len(names))}
        id2label = {idx: name for idx, name in enumerate(names)}
        return label2id, id2label
    labels = dataset.unique(label_field)
    label2id = {str(label): idx for idx, label in enumerate(sorted(labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _attach_labels(dataset: Any, label_field: str | None, label2id: dict[str, int]) -> Any:
    if label_field is None:
        raise ValueError("label_field must be set for supervised training.")

    def map_labels(batch: dict[str, Any]) -> dict[str, Any]:
        labels = batch[label_field]
        if isinstance(labels, list):
            return {"labels": [label2id[str(label)] for label in labels]}
        return {"labels": label2id[str(labels)]}

    return dataset.map(map_labels, batched=True)


def _tokenize_batch(
    batch: dict[str, Any],
    config: DatasetConfig,
    tokenizer: CtokHFTokenizer,
    max_length: int,
) -> dict[str, Any]:
    if config.sequence_field:
        sequences = batch.get(config.sequence_field)
        if sequences is None:
            raise KeyError(f"Missing sequence_field '{config.sequence_field}'")
        texts = [config.sequence_delimiter.join(map(str, seq)) for seq in sequences]
    elif config.text_fields:
        field_batches = []
        for field in config.text_fields:
            values = batch.get(field)
            if values is None:
                raise KeyError(f"Missing text_fields entry '{field}'")
            field_batches.append(["" if value is None else str(value) for value in values])
        texts = []
        for values in zip(*field_batches):
            parts = [value for value in values if value]
            texts.append(config.field_delimiter.join(parts))
    else:
        texts = batch.get(config.text_field)
        if texts is None:
            raise KeyError(f"Missing text_field '{config.text_field}'")
        texts = [str(text) for text in texts]
    output = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    if "labels" in batch:
        output["labels"] = batch["labels"]
    return output


def _compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    logits, labels = eval_pred
    scores = _to_list(logits)
    gold = _to_list(labels)

    if not scores:
        return {}

    num_labels = len(scores[0]) if isinstance(scores[0], list) else 1
    preds = [_argmax(row) for row in scores] if num_labels > 1 else [1 if s >= 0 else 0 for s in scores]

    metrics = {"accuracy": _accuracy(gold, preds)}
    if num_labels == 2:
        probs = [_softmax(row)[1] for row in scores]
        metrics["f1"] = _binary_f1(gold, preds)
        metrics["auroc"] = _binary_auc(gold, probs)
    else:
        metrics["f1_macro"] = _macro_f1(gold, preds, num_labels)
    return metrics


def _to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _argmax(row: list[float]) -> int:
    return max(range(len(row)), key=lambda idx: row[idx])


def _softmax(row: list[float]) -> list[float]:
    max_val = max(row)
    exp_vals = [math.exp(val - max_val) for val in row]
    denom = sum(exp_vals) or 1.0
    return [val / denom for val in exp_vals]


def _accuracy(gold: list[int], preds: list[int]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, preds) if g == p)
    return correct / len(gold)


def _binary_f1(gold: list[int], preds: list[int]) -> float:
    tp = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, preds) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, preds) if g == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro_f1(gold: list[int], preds: list[int], num_labels: int) -> float:
    scores = []
    for label in range(num_labels):
        tp = sum(1 for g, p in zip(gold, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, preds) if g == label and p != label)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _binary_auc(labels: list[int], scores: list[float]) -> float:
    paired = list(zip(scores, labels))
    paired.sort(key=lambda item: item[0])

    pos = sum(1 for _, label in paired if label == 1)
    neg = len(paired) - pos
    if pos == 0 or neg == 0:
        return 0.0

    rank = 1
    sum_pos_ranks = 0.0
    idx = 0
    while idx < len(paired):
        score = paired[idx][0]
        jdx = idx
        while jdx < len(paired) and paired[jdx][0] == score:
            jdx += 1
        avg_rank = (rank + (rank + (jdx - idx) - 1)) / 2
        for _, label in paired[idx:jdx]:
            if label == 1:
                sum_pos_ranks += avg_rank
        rank += jdx - idx
        idx = jdx

    return (sum_pos_ranks - pos * (pos + 1) / 2) / (pos * neg)
