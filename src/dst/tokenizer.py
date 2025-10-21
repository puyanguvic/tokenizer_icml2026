from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.token_id: Optional[int] = None


def _insert_into_trie(root: TrieNode, token: str, token_id: int) -> None:
    node = root
    for char in token:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.token_id = token_id


def _build_trie(tokens: Sequence[str]) -> TrieNode:
    root = TrieNode()
    for token_id, token in enumerate(tokens):
        _insert_into_trie(root, token, token_id)
    return root


@dataclass
class DFSTState:
    transitions: Dict[str, int]
    emit: Optional[int] = None


def _collect_deterministic_transitions(root: TrieNode) -> List[DFSTState]:
    """
    Flatten the trie into deterministic transitions suitable for serialization.

    Each state corresponds to a node in the trie and stores outgoing edges.
    """
    states: List[DFSTState] = []
    queue: List[TrieNode] = [root]
    index_map: Dict[int, int] = {id(root): 0}

    while queue:
        node = queue.pop(0)
        state_index = index_map[id(node)]
        # Extend states list if necessary.
        if state_index >= len(states):
            states.append(DFSTState(transitions={}, emit=node.token_id))
        else:
            states[state_index].emit = node.token_id

        for char, child in node.children.items():
            if id(child) not in index_map:
                index_map[id(child)] = len(states)
                states.append(DFSTState(transitions={}, emit=child.token_id))
                queue.append(child)
            states[state_index].transitions[char] = index_map[id(child)]
    return states


class DSTTokenizer:
    """
    Deterministic tokenizer with guaranteed invertibility via longest-prefix decoding.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        special_tokens: Sequence[str],
        fallback_tokens: Sequence[str],
        normalizer: Callable[[str], str],
        candidate_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.special_tokens = list(special_tokens)
        self.domain_tokens = list(tokens)
        self.fallback_tokens = list(fallback_tokens)
        self.normalizer = normalizer
        self.candidate_metadata = candidate_metadata or {}

        self.tokens: List[str] = (
            self.special_tokens + self.domain_tokens + self.fallback_tokens
        )
        self.token_to_id: Dict[str, int] = {
            token: idx for idx, token in enumerate(self.tokens)
        }
        self.root = _build_trie(self.domain_tokens + self.fallback_tokens)
        self.dfst_states = _collect_deterministic_transitions(self.root)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        tokens_with_offsets = self.encode_with_offsets(text)
        return [self.token_to_id[token] for token, _, _ in tokens_with_offsets]

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.tokens):
                raise ValueError(f"Unknown token id: {token_id}")
            token = self.tokens[token_id]
            if token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def encode_to_tokens(self, text: str) -> List[str]:
        return [token for token, _, _ in self.encode_with_offsets(text)]

    def _tokenize_with_offsets(self, normalized: str) -> List[Tuple[str, int, int]]:
        if not normalized:
            return []

        tokens: List[Tuple[str, int, int]] = []
        length = len(normalized)
        index = 0
        domain_and_fallback = self.domain_tokens + self.fallback_tokens

        while index < length:
            node = self.root
            longest_match_token: Optional[str] = None
            longest_match_length = 0
            lookahead = index

            while lookahead < length:
                char = normalized[lookahead]
                if char not in node.children:
                    break
                node = node.children[char]
                lookahead += 1
                if node.token_id is not None:
                    token = domain_and_fallback[node.token_id]
                    longest_match_token = token
                    longest_match_length = lookahead - index

            if longest_match_token is None:
                longest_match_token = normalized[index]
                longest_match_length = 1

            tokens.append(
                (longest_match_token, index, index + longest_match_length)
            )
            index += longest_match_length

        return tokens

    def encode_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        normalized = self.normalizer(text)
        return self._tokenize_with_offsets(normalized)

    def _encode_to_tokens(self, normalized: str) -> List[str]:
        return [token for token, _, _ in self._tokenize_with_offsets(normalized)]

    # ------------------------------------------------------------------
    # Validation & Metrics
    # ------------------------------------------------------------------
    def validate_consistency(self, samples: Iterable[str]) -> float:
        total = 0
        consistent = 0
        for text in samples:
            total += 1
            encoded = self.encode(text)
            decoded = self.decode(encoded)
            if decoded == self.normalizer(text):
                consistent += 1
        return consistent / total if total else 1.0

    def average_tokens_per_char(self, samples: Iterable[str]) -> float:
        total_chars = 0
        total_tokens = 0
        for text in samples:
            normalized = self.normalizer(text)
            total_chars += len(normalized)
            total_tokens += len(self._tokenize_with_offsets(normalized))
        return total_tokens / total_chars if total_chars else 0.0

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_serializable(self) -> Dict[str, object]:
        return {
            "type": "dst",
            "version": "0.1.0",
            "special_tokens": self.special_tokens,
            "domain_tokens": self.domain_tokens,
            "fallback_tokens": self.fallback_tokens,
            "candidate_metadata": self.candidate_metadata,
            "transducer": [
                {
                    "emit": state.emit,
                    "transitions": state.transitions,
                }
                for state in self.dfst_states
            ],
        }

    def _build_hf_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        vocab = {token: idx for idx, token in enumerate(self.tokens)}
        unk_token = "[UNK]" if "[UNK]" in self.token_to_id else None
        model = models.WordLevel(vocab=vocab, unk_token=unk_token)
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        if "[CLS]" in self.token_to_id and "[SEP]" in self.token_to_id:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.token_to_id["[CLS]"]),
                    ("[SEP]", self.token_to_id["[SEP]"]),
                ],
            )

        return tokenizer

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        hf_tokenizer = self._build_hf_tokenizer()
        hf_tokenizer.save(os.path.join(directory, "tokenizer.json"))

        with open(os.path.join(directory, "dst_tokenizer.json"), "w", encoding="utf-8") as fp:
            json.dump(self.to_serializable(), fp, ensure_ascii=False, indent=2)

        with open(os.path.join(directory, "vocab.txt"), "w", encoding="utf-8") as fp:
            for token in self.tokens:
                fp.write(token + "\n")

        special_map = {}
        if "[CLS]" in self.token_to_id:
            special_map["cls_token"] = "[CLS]"
        if "[SEP]" in self.token_to_id:
            special_map["sep_token"] = "[SEP]"
        if "[PAD]" in self.token_to_id:
            special_map["pad_token"] = "[PAD]"
        if "[MASK]" in self.token_to_id:
            special_map["mask_token"] = "[MASK]"
        if "[UNK]" in self.token_to_id:
            special_map["unk_token"] = "[UNK]"

        if special_map:
            with open(
                os.path.join(directory, "special_tokens_map.json"), "w", encoding="utf-8"
            ) as fp:
                json.dump(special_map, fp, indent=2)

        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "backend_tokenizer_file": "tokenizer.json",
        }
        if special_map:
            tokenizer_config["special_tokens_map_file"] = "special_tokens_map.json"
        with open(
            os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(tokenizer_config, fp, indent=2)

    # Convenience API ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)
