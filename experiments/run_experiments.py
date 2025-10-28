from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:  # pragma: no cover - optional dependency
    from datasets import Dataset  # type: ignore
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore
    load_dataset = None  # type: ignore

from http_tokenizer import HTTP_GRAMMAR_PATTERNS, http_clean_line  # noqa: E402
from metrics import (  # noqa: E402
    average_token_length,
    compression_ratio,
    round_trip_accuracy,
)
from metrics import _baseline_token_count  # type: ignore  # noqa: E402
from dst.pipeline import CandidateExtractorConfig, ScoreWeights, build_dst_tokenizer  # noqa: E402


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _default_normalizer(text: str) -> str:
    return text.strip()


def _config_normalizer(text: str) -> str:
    lines = [line.rstrip().replace("\t", "    ") for line in text.splitlines()]
    return "\n".join(lines)


def _code_normalizer(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def _bio_normalizer(text: str) -> str:
    return "".join(text.split()).upper()


CONFIG_GRAMMAR_PATTERNS = [
    r"[A-Za-z0-9_\-]+\s*:\s*[^\n#]+",  # YAML-style key: value
    r"-\s+[^\s]+",  # list entries
    r"\{[^{}]+\}",  # inline JSON objects
    r"\[[^\[\]]+\]",  # inline arrays
    r"\$\{[A-Za-z0-9_\.\-:]+\}",  # templated placeholders
]

CODE_GRAMMAR_PATTERNS = [
    r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",  # function definitions
    r"class\s+[A-Za-z_][A-Za-z0-9_]*",  # class definitions
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*",  # assignments
    r"[A-Za-z_][A-Za-z0-9_]*\s*\(",  # function calls
    r"import\s+[A-Za-z0-9_\.,\s]+",  # import statements
    r"\"[^\"]+\"|\'[^\']+\'",  # string literals
]

BIO_GRAMMAR_PATTERNS = [
    r">[^\n]+",  # FASTA headers
    r"[ACGTN]{6,}",  # nucleotide runs
    r"[A-Z]{3}\s+[A-Z]{3}\s+[A-Z]{3}",  # codon triplets
]


@dataclass
class DomainRecipe:
    name: str
    description: str
    normalizer: Callable[[str], str]
    grammar_patterns: Sequence[str]
    preserve_case: bool = True
    vocab_size: int = 32000


DOMAIN_RECIPES: Dict[str, DomainRecipe] = {
    "protocol": DomainRecipe(
        name="protocol",
        description="HTTP / protocol traces with percent-decoding normalization.",
        normalizer=http_clean_line,
        grammar_patterns=HTTP_GRAMMAR_PATTERNS,
        preserve_case=True,
        vocab_size=32000,
    ),
    "config": DomainRecipe(
        name="config",
        description="YAML / JSON configuration snippets.",
        normalizer=_config_normalizer,
        grammar_patterns=CONFIG_GRAMMAR_PATTERNS,
        preserve_case=False,
        vocab_size=24000,
    ),
    "code": DomainRecipe(
        name="code",
        description="Source code functions and class definitions.",
        normalizer=_code_normalizer,
        grammar_patterns=CODE_GRAMMAR_PATTERNS,
        preserve_case=True,
        vocab_size=32000,
    ),
    "bio": DomainRecipe(
        name="bio",
        description="Biosequence data (DNA/RNA) with FASTA headers.",
        normalizer=_bio_normalizer,
        grammar_patterns=BIO_GRAMMAR_PATTERNS,
        preserve_case=False,
        vocab_size=20000,
    ),
}


def _split_samples(samples: Sequence[str], eval_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    cutoff = max(1, int(len(samples) * (1.0 - eval_ratio)))
    train_indices = indices[:cutoff]
    eval_indices = indices[cutoff:]
    train = [samples[i] for i in train_indices]
    eval_set = [samples[i] for i in eval_indices] or train[:max(1, len(train) // 5)]
    return train, eval_set


def _load_hf_samples(
    name: str,
    split: str,
    text_field: str,
    limit: Optional[int],
    streaming: bool = False,
) -> List[str]:
    if load_dataset is None:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'datasets' package is required to load Hugging Face datasets.")

    dataset = load_dataset(name, split=split, streaming=streaming)

    def _extract(example):
        value = example
        for part in text_field.split("."):
            value = value[part]
        return "" if value is None else str(value)

    samples: List[str] = []
    if streaming:
        assert limit is not None, "Streaming mode requires an explicit limit."
        for example in dataset:  # type: ignore[attr-defined]
            samples.append(_extract(example))
            if len(samples) >= limit:
                break
    else:
        assert isinstance(dataset, Dataset)
        effective_limit = min(limit or len(dataset), len(dataset))
        for example in dataset.shuffle(seed=13).select(range(effective_limit)):  # type: ignore[attr-defined]
            samples.append(_extract(example))
    return samples


def _load_local_samples(path: Path, limit: Optional[int]) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    if path.suffix.lower() in {".txt", ".log"}:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
    elif path.suffix.lower() in {".jsonl", ".json"}:
        import json  # local import for optional dependency

        lines = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                value = record.get("text") or record.get("content") or record.get("value")
                if value is None:
                    continue
                lines.append(str(value))
    else:
        raise ValueError(f"Unsupported corpus format: {path.suffix}")

    return lines[:limit] if limit is not None else lines


def _normalize_samples(samples: Iterable[str], normalizer: Callable[[str], str], preserve_case: bool) -> List[str]:
    normalized = []
    for sample in samples:
        norm = normalizer(sample)
        norm = norm if preserve_case else norm.lower()
        normalized.append(norm)
    return normalized


def _evaluate_baseline_tokenizer(tokenizer, samples: Sequence[str], baseline_counts: Sequence[int]) -> Dict[str, float]:
    total_tokens = 0
    consistent = 0
    for sample in samples:
        encoded = tokenizer.encode(sample)
        total_tokens += len(encoded.ids)
        try:
            decoded = tokenizer.decode(encoded.ids)
        except Exception:  # pragma: no cover - conservative decode guard
            decoded = ""
        if decoded == sample:
            consistent += 1
    denom = sum(baseline_counts) or 1
    return {
        "round_trip": consistent / len(samples) if samples else 1.0,
        "compression_ratio": total_tokens / denom,
        "avg_tokens_per_sample": total_tokens / len(samples) if samples else 0.0,
    }


def _train_baseline(kind: str, samples: Sequence[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if kind == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.train_from_iterator(samples, trainer=trainer)
        tokenizer.decoder = decoders.BPEDecoder()
        return tokenizer

    if kind == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.train_from_iterator(samples, trainer=trainer)
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        return tokenizer

    if kind == "bytebpe":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(samples, trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    raise ValueError(f"Unknown baseline tokenizer: {kind}")


def run_experiment(
    domain: str,
    samples: Sequence[str],
    recipe: DomainRecipe,
    output_dir: Optional[Path],
    baselines: Sequence[str],
) -> Dict[str, object]:
    train_samples, eval_samples = _split_samples(samples, eval_ratio=0.2, seed=13)
    if not train_samples or not eval_samples:
        raise ValueError("Insufficient samples for experiment. Provide more data.")

    normalized_train = _normalize_samples(train_samples, recipe.normalizer, recipe.preserve_case)
    normalized_eval = _normalize_samples(eval_samples, recipe.normalizer, recipe.preserve_case)
    baseline_counts = [_baseline_token_count(text) for text in normalized_eval]

    config = CandidateExtractorConfig(
        max_vocab=recipe.vocab_size,
        grammar_patterns=recipe.grammar_patterns,
        preserve_case=recipe.preserve_case,
        special_tokens=SPECIAL_TOKENS,
        weights=ScoreWeights(),
    )

    tokenizer = build_dst_tokenizer(
        corpus=train_samples,
        normalizer=recipe.normalizer,
        config=config,
        save_dir=str(output_dir / domain) if output_dir else None,
    )

    dst_results = {
        "round_trip": round_trip_accuracy(tokenizer, eval_samples),
        "compression_ratio": compression_ratio(tokenizer, eval_samples),
        "avg_token_length": average_token_length(tokenizer),
    }

    baseline_results: Dict[str, Dict[str, float]] = {}
    if baselines:
        for kind in baselines:
            baseline_tokenizer = _train_baseline(kind, normalized_train, vocab_size=recipe.vocab_size)
            baseline_results[kind] = _evaluate_baseline_tokenizer(
                baseline_tokenizer,
                normalized_eval,
                baseline_counts,
            )

    results: Dict[str, object] = {
        "domain": domain,
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "dst": dst_results,
        "baselines": baseline_results,
        "description": recipe.description,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{domain}_results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DST experiments across structured domains.")
    parser.add_argument("--domain", required=True, choices=DOMAIN_RECIPES.keys(), help="Domain key to evaluate.")
    parser.add_argument(
        "--dataset",
        help="Hugging Face dataset spec in the form 'name[:config]/split'. Example: 'bigcode/the-stack:python/train'.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Local corpus path (.txt or .jsonl) containing one record per line.",
    )
    parser.add_argument("--text-field", default="text", help="Field name for datasets with structured records.")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum number of samples to load.")
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip training baseline tokenizers (BPE, WordPiece, Byte-BPE).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store tokenizer artifacts and JSON summaries.",
    )

    args = parser.parse_args()

    recipe = DOMAIN_RECIPES[args.domain]
    samples: List[str]

    if args.dataset:
        if "/" not in args.dataset:
            raise ValueError("Dataset spec must include a split, e.g. 'dataset/train'.")
        name_part, split = args.dataset.split("/", maxsplit=1)
        if ":" in name_part:
            name, config = name_part.split(":", maxsplit=1)
            dataset_name = name
            dataset_kwargs = {"name": config}
        else:
            dataset_name = name_part
            dataset_kwargs = {}
        samples = _load_hf_samples(
            name=dataset_name,
            split=split,
            text_field=args.text_field,
            limit=args.limit,
            streaming=False,
            **dataset_kwargs,
        )
    elif args.corpus:
        samples = _load_local_samples(args.corpus, limit=args.limit)
    else:
        raise ValueError("Provide either --dataset or --corpus to supply experimental data.")

    if not samples:
        raise ValueError("Loaded corpus is empty. Check dataset spec and text field.")

    baselines = [] if args.no_baselines else ["bpe", "wordpiece", "bytebpe"]
    results = run_experiment(
        domain=args.domain,
        samples=samples,
        recipe=recipe,
        output_dir=args.output_dir,
        baselines=baselines,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
