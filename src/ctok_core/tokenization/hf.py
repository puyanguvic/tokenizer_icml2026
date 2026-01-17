"""Hugging Face Transformers tokenizer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ctok_core.tokenization.io import load_tokenizer, save_tokenizer
from ctok_core.tokenization.rules import RuleSet
from ctok_core.tokenization.tokenizer import CtokTokenizer
from ctok_core.tokenization.vocab import Vocabulary
from ctok_core.utils.serialization import read_json

try:
    from transformers import PreTrainedTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install 'transformers' to use CtokHFTokenizer.") from exc

_HF_TO_INTERNAL = {
    "pad_token": "pad",
    "unk_token": "unk",
    "cls_token": "cls",
    "sep_token": "sep",
    "bos_token": "bos",
    "eos_token": "eos",
    "mask_token": "mask",
}


class CtokHFTokenizer(PreTrainedTokenizer):
    """Transformers-compatible wrapper around CtokTokenizer."""

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "rules_file": "rules.json",
        "manifest_file": "manifest.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Vocabulary,
        rules: RuleSet | None = None,
        **kwargs: Any,
    ) -> None:
        raw_special = kwargs.pop("special_tokens", None)
        special_tokens = dict(raw_special) if raw_special else {}
        padding_side = kwargs.pop("padding_side", "right")
        truncation_side = kwargs.pop("truncation_side", "right")
        boundary_mode = kwargs.pop("boundary_mode", "none")
        boundary_chars = kwargs.pop("boundary_chars", None)

        hf_token_kwargs = {}
        for hf_key, internal in _HF_TO_INTERNAL.items():
            if hf_key in kwargs:
                token = kwargs.pop(hf_key)
                if token is not None:
                    hf_token_kwargs[hf_key] = token
                    special_tokens.setdefault(internal, token)

        self._ctok = CtokTokenizer(
            vocab=vocab,
            rules=rules,
            special_tokens=special_tokens or None,
            padding_side=padding_side,
            truncation_side=truncation_side,
            boundary_mode=boundary_mode,
            boundary_chars=boundary_chars,
        )

        super().__init__(
            **hf_token_kwargs,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_input_names=list(self.model_input_names),
            **kwargs,
        )
        self.boundary_mode = boundary_mode
        self.boundary_chars = boundary_chars
        self._auto_map = {"AutoTokenizer": "ctok_core.tokenization.hf.CtokHFTokenizer"}
        self.init_kwargs["boundary_mode"] = boundary_mode
        self.init_kwargs["boundary_chars"] = boundary_chars

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return self._ctok.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self._ctok.get_vocab()

    def _tokenize(self, text: str) -> list[str]:
        return self._ctok.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._ctok.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._ctok.id_to_token(index)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return self._ctok.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        special_ids = set(self._ctok.vocab.special_ids().values())
        if already_has_special_tokens:
            return [1 if token_id in special_ids else 0 for token_id in token_ids_0]

        if token_ids_1 is None:
            built = self.build_inputs_with_special_tokens(token_ids_0)
        else:
            built = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        return [1 if token_id in special_ids else 0 for token_id in built]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return [0] * len(self.build_inputs_with_special_tokens(token_ids_0))
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        rules = self._ctok.rules or RuleSet.from_vocab(self._ctok.vocab)
        save_tokenizer(output_dir, self._ctok.vocab, rules)

        vocab_path = output_dir / "vocab.json"
        rules_path = output_dir / "rules.json"
        manifest_path = output_dir / "manifest.json"
        return str(vocab_path), str(rules_path), str(manifest_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any) -> "CtokHFTokenizer":
        artifact_dir = Path(pretrained_model_name_or_path)
        vocab, rules, _manifest = load_tokenizer(artifact_dir)
        config_path = artifact_dir / "tokenizer_config.json"
        if config_path.exists():
            config = read_json(config_path)
            special_tokens = dict(config.get("special_tokens") or {})
            kwargs.setdefault("special_tokens", special_tokens or None)
            for hf_key in _HF_TO_INTERNAL:
                if config.get(hf_key) and hf_key not in kwargs:
                    kwargs[hf_key] = config[hf_key]
            if "padding_side" not in kwargs:
                kwargs["padding_side"] = config.get("padding_side", "right")
            if "truncation_side" not in kwargs:
                kwargs["truncation_side"] = config.get("truncation_side", "right")
            if "boundary_mode" not in kwargs:
                kwargs["boundary_mode"] = config.get("boundary_mode", "none")
            if "boundary_chars" not in kwargs:
                kwargs["boundary_chars"] = config.get("boundary_chars")
        return cls(vocab=vocab, rules=rules, **kwargs)
