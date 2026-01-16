"""High-level tokenizer interface for model integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctok.tokenization.io import load_tokenizer, save_tokenizer
from ctok.tokenization.rules import RuleSet
from ctok.tokenization.boundary import DEFAULT_BOUNDARY_CHARS, normalize_boundary_chars
from ctok.tokenization.runtime import BoundaryAwareTokenizerRuntime, TokenizerRuntime
from ctok.tokenization.vocab import Vocabulary
from ctok.utils.serialization import read_json, write_json


@dataclass
class Encoding:
    input_ids: list[int]
    attention_mask: list[int] | None = None
    token_type_ids: list[int] | None = None


class CtokTokenizer:
    """Tokenizer wrapper with a Transformers-style interface."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Vocabulary,
        rules: RuleSet | None = None,
        special_tokens: dict[str, str] | None = None,
        padding_side: str = "right",
        truncation_side: str = "right",
        boundary_mode: str = "none",
        boundary_chars: list[str] | set[str] | str | None = None,
    ) -> None:
        if special_tokens:
            vocab = vocab.with_special_tokens(special_tokens)
        self._vocab = vocab
        self._rules = rules
        if boundary_mode not in {"none", "aware"}:
            raise ValueError("boundary_mode must be 'none' or 'aware'")
        self.boundary_mode = boundary_mode
        self.boundary_chars = normalize_boundary_chars(boundary_chars)
        if self.boundary_mode == "aware":
            chars = self.boundary_chars or DEFAULT_BOUNDARY_CHARS
            self._runtime = BoundaryAwareTokenizerRuntime(vocab, chars)
        else:
            self._runtime = TokenizerRuntime(vocab)
        self._token_to_id = vocab.token_to_id()
        self._special_tokens = dict(vocab.special_tokens)
        self.padding_side = padding_side
        self.truncation_side = truncation_side

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab.tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def special_tokens_map(self) -> dict[str, str]:
        return dict(self._special_tokens)

    def token_to_id(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            return self._unk_id_or_raise(token)
        return token_id

    def id_to_token(self, token_id: int) -> str:
        return self._vocab.token_for(token_id)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.token_to_id(tokens)
        return [self.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self.id_to_token(ids)
        return [self.id_to_token(token_id) for token_id in ids]

    def tokenize(self, text: str) -> list[str]:
        return [self.id_to_token(idx) for idx in self.encode(text)]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = self._runtime.encode(text)
        if add_special_tokens:
            ids = self._add_special_tokens(ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._runtime.decode(ids)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch]

    def add_special_tokens(self, special_tokens: dict[str, str]) -> int:
        tokens = list(self._vocab.tokens)
        merged = dict(self._vocab.special_tokens)
        added = 0
        for name, token in special_tokens.items():
            if token not in tokens:
                tokens.append(token)
                added += 1
            merged[name] = token
        self._vocab = Vocabulary(tokens=tokens, special_tokens=merged)
        self._rules = RuleSet.from_vocab(self._vocab)
        if self.boundary_mode == "aware":
            chars = self.boundary_chars or DEFAULT_BOUNDARY_CHARS
            self._runtime = BoundaryAwareTokenizerRuntime(self._vocab, chars)
        else:
            self._runtime = TokenizerRuntime(self._vocab)
        self._token_to_id = self._vocab.token_to_id()
        self._special_tokens = dict(self._vocab.special_tokens)
        return added

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | list[str] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> dict[str, Any]:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        pairs: list[str] | None = None
        if text_pair is not None:
            if isinstance(text_pair, str):
                pairs = [text_pair]
            else:
                pairs = list(text_pair)
            if len(pairs) != len(texts):
                raise ValueError("text_pair must be the same length as text")

        encodings = []
        for idx, item in enumerate(texts):
            ids = self._runtime.encode(item)
            pair_ids = None
            if pairs is not None:
                pair_ids = self._runtime.encode(pairs[idx])
            if add_special_tokens:
                ids = self.build_inputs_with_special_tokens(ids, pair_ids)
            elif pair_ids is not None:
                ids = ids + pair_ids
            encodings.append(Encoding(input_ids=ids))

        if truncation:
            encodings = [self._truncate(enc, max_length) for enc in encodings]

        if padding:
            encodings = self._pad_batch(encodings, padding, max_length)

        result = self._encoding_to_output(encodings, return_attention_mask, return_token_type_ids)
        return self._maybe_convert_tensors(result, return_tensors)

    def encode_plus(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.__call__(*args, **kwargs)

    def batch_encode_plus(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.__call__(*args, **kwargs)

    def pad(
        self,
        encoded_inputs: dict[str, Any] | list[dict[str, Any]],
        padding: bool | str = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(encoded_inputs, dict):
            input_ids = encoded_inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("encoded_inputs must contain input_ids")
            if input_ids and isinstance(input_ids[0], int):
                encodings = [Encoding(input_ids=input_ids)]
            else:
                encodings = [Encoding(input_ids=ids) for ids in input_ids]
        else:
            encodings = [Encoding(input_ids=item["input_ids"]) for item in encoded_inputs]

        if padding:
            encodings = self._pad_batch(encodings, padding, max_length)

        output = self._encoding_to_output(encodings, return_attention_mask=True, return_token_type_ids=False)
        return self._maybe_convert_tensors(output, return_tensors)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        ids = list(token_ids_0)
        if token_ids_1 is None:
            return self._wrap_with_special_tokens(ids)
        return self._wrap_pair_with_special_tokens(ids, list(token_ids_1))

    def save_pretrained(self, output_dir: str | Path, metadata: dict[str, object] | None = None) -> list[Path]:
        output_path = Path(output_dir)
        save_tokenizer(output_path, self._vocab, self._rules or RuleSet.from_vocab(self._vocab), metadata=metadata)
        config_path = output_path / "tokenizer_config.json"
        write_json(
            config_path,
            {
                "tokenizer_class": "CtokTokenizer",
                "special_tokens": dict(self._special_tokens),
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "boundary_mode": self.boundary_mode,
                "boundary_chars": sorted(self.boundary_chars) if self.boundary_chars else None,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
                "model_input_names": list(self.model_input_names),
            },
        )
        return [output_path / "vocab.json", output_path / "rules.json", output_path / "manifest.json", config_path]

    @classmethod
    def from_pretrained(cls, artifact_dir: str | Path) -> "CtokTokenizer":
        vocab, rules, _manifest = load_tokenizer(Path(artifact_dir))
        config_path = Path(artifact_dir) / "tokenizer_config.json"
        kwargs: dict[str, Any] = {}
        if config_path.exists():
            config = read_json(config_path)
            special_tokens = dict(config.get("special_tokens") or {})
            for hf_key, internal in {
                "pad_token": "pad",
                "unk_token": "unk",
                "cls_token": "cls",
                "sep_token": "sep",
                "bos_token": "bos",
                "eos_token": "eos",
                "mask_token": "mask",
            }.items():
                value = config.get(hf_key)
                if value:
                    special_tokens.setdefault(internal, value)
            kwargs["special_tokens"] = special_tokens or None
            kwargs["padding_side"] = config.get("padding_side", "right")
            kwargs["truncation_side"] = config.get("truncation_side", "right")
            kwargs["boundary_mode"] = config.get("boundary_mode", "none")
            kwargs["boundary_chars"] = config.get("boundary_chars")
        return cls(vocab=vocab, rules=rules, **kwargs)

    def _wrap_with_special_tokens(self, ids: list[int]) -> list[int]:
        cls_id = self._special_token_id("cls")
        sep_id = self._special_token_id("sep")
        bos_id = self._special_token_id("bos")
        eos_id = self._special_token_id("eos")

        if cls_id is not None:
            ids = [cls_id] + ids
        elif bos_id is not None:
            ids = [bos_id] + ids

        if sep_id is not None:
            ids = ids + [sep_id]
        elif eos_id is not None:
            ids = ids + [eos_id]
        return ids

    def _add_special_tokens(self, ids: list[int]) -> list[int]:
        return self._wrap_with_special_tokens(list(ids))

    def _wrap_pair_with_special_tokens(self, ids: list[int], pair_ids: list[int]) -> list[int]:
        cls_id = self._special_token_id("cls")
        sep_id = self._special_token_id("sep")
        bos_id = self._special_token_id("bos")
        eos_id = self._special_token_id("eos")

        if cls_id is not None:
            tokens = [cls_id] + ids
        elif bos_id is not None:
            tokens = [bos_id] + ids
        else:
            tokens = list(ids)

        if sep_id is not None:
            tokens.append(sep_id)
            tokens.append(sep_id)
            tokens += pair_ids
            tokens.append(sep_id)
            return tokens

        if eos_id is not None:
            tokens.append(eos_id)

        tokens += pair_ids

        if eos_id is not None:
            tokens.append(eos_id)
        return tokens

    def _truncate(self, encoding: Encoding, max_length: int | None) -> Encoding:
        if max_length is None:
            return encoding
        input_ids = encoding.input_ids
        if len(input_ids) <= max_length:
            return encoding
        if self.truncation_side == "left":
            truncated = input_ids[-max_length:]
        else:
            truncated = input_ids[:max_length]
        return Encoding(input_ids=truncated)

    def _pad_batch(
        self,
        encodings: list[Encoding],
        padding: bool | str,
        max_length: int | None,
    ) -> list[Encoding]:
        if padding is True:
            target_length = max(len(enc.input_ids) for enc in encodings)
        elif padding == "longest":
            target_length = max(len(enc.input_ids) for enc in encodings)
        elif padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'")
            target_length = max_length
        else:
            return encodings

        pad_id = self.pad_token_id
        if pad_id is None:
            raise ValueError("pad_token is not set; cannot pad")

        padded = []
        for enc in encodings:
            ids = list(enc.input_ids)
            pad_len = target_length - len(ids)
            if pad_len <= 0:
                padded.append(enc)
                continue
            if self.padding_side == "left":
                ids = [pad_id] * pad_len + ids
            else:
                ids = ids + [pad_id] * pad_len
            padded.append(Encoding(input_ids=ids))
        return padded

    def _encoding_to_output(
        self,
        encodings: list[Encoding],
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, Any]:
        input_ids = [enc.input_ids for enc in encodings]
        output: dict[str, Any] = {"input_ids": input_ids}

        if return_attention_mask:
            pad_id = self.pad_token_id
            if pad_id is None:
                attention = [[1] * len(ids) for ids in input_ids]
            else:
                attention = [[0 if token == pad_id else 1 for token in ids] for ids in input_ids]
            output["attention_mask"] = attention

        if return_token_type_ids:
            output["token_type_ids"] = [[0] * len(ids) for ids in input_ids]

        if len(input_ids) == 1:
            return {key: value[0] for key, value in output.items()}
        return output

    def _maybe_convert_tensors(self, payload: dict[str, Any], return_tensors: str | None) -> dict[str, Any]:
        if return_tensors is None:
            return payload
        if return_tensors == "np":
            try:
                import numpy as np  # type: ignore
            except ImportError as exc:
                raise ImportError("Install numpy to use return_tensors='np'") from exc
            return {key: np.asarray(value) for key, value in payload.items()}
        if return_tensors == "pt":
            try:
                import torch  # type: ignore
            except ImportError as exc:
                raise ImportError("Install torch to use return_tensors='pt'") from exc
            return {key: torch.tensor(value) for key, value in payload.items()}
        raise ValueError(f"Unsupported return_tensors: {return_tensors}")

    def _special_token_id(self, name: str) -> int | None:
        token = self._special_tokens.get(name)
        if token is None:
            return None
        return self._token_to_id.get(token)

    def _unk_id_or_raise(self, token: str) -> int:
        unk_id = self._special_token_id("unk")
        if unk_id is not None:
            return unk_id
        raise KeyError(f"Token not in vocabulary: {token}")

    @property
    def pad_token(self) -> str | None:
        return self._special_tokens.get("pad")

    @property
    def pad_token_id(self) -> int | None:
        return self._special_token_id("pad")

    @property
    def unk_token(self) -> str | None:
        return self._special_tokens.get("unk")

    @property
    def unk_token_id(self) -> int | None:
        return self._special_token_id("unk")

    @property
    def cls_token(self) -> str | None:
        return self._special_tokens.get("cls")

    @property
    def sep_token(self) -> str | None:
        return self._special_tokens.get("sep")

    @property
    def bos_token(self) -> str | None:
        return self._special_tokens.get("bos")

    @property
    def eos_token(self) -> str | None:
        return self._special_tokens.get("eos")

    @property
    def mask_token(self) -> str | None:
        return self._special_tokens.get("mask")
