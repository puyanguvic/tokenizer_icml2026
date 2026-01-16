"""Serialization for tokenizer artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ctok.tokenization.rules import RuleSet
from ctok.tokenization.vocab import Vocabulary
from ctok.utils.hashing import sha256_file
from ctok.utils.serialization import read_json, write_json


def save_tokenizer(
    output_dir: Path,
    vocab: Vocabulary,
    rules: RuleSet,
    metadata: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    rules_path = output_dir / "rules.json"
    manifest_path = output_dir / "manifest.json"

    write_json(vocab_path, vocab.to_dict())
    write_json(rules_path, rules.to_dict())

    manifest = {
        "format_version": 1,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "vocab_size": len(vocab.tokens),
        "metadata": metadata or {},
        "files": {
            "vocab.json": sha256_file(vocab_path),
            "rules.json": sha256_file(rules_path),
        },
    }
    write_json(manifest_path, manifest)


def load_tokenizer(
    artifact_dir: Path,
) -> tuple[Vocabulary, RuleSet, dict[str, object]]:
    vocab_path = artifact_dir / "vocab.json"
    rules_path = artifact_dir / "rules.json"
    manifest_path = artifact_dir / "manifest.json"

    vocab = Vocabulary.from_dict(read_json(vocab_path))
    rules = RuleSet.from_dict(read_json(rules_path))
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return vocab, rules, manifest
