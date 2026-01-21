from __future__ import annotations

import json

CIT_ARTIFACT_SCHEMA_VERSION = "cit_artifact.v1"

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence

from ..interface.contract import Contract, ContractConfig
from .compiler import CompiledMatcher


@dataclass
class CITArtifact:
    """In-memory representation of a CIT tokenizer artifact."""

    meta: Dict[str, Any]
    vocab: Dict[str, int]
    matcher: CompiledMatcher
    contract: ContractConfig
    special_tokens: Sequence[str]

    @property
    def id_to_token(self) -> List[str]:
        inv = [""] * (max(self.vocab.values()) + 1)
        for t, i in self.vocab.items():
            inv[i] = t
        return inv

    def dumps(self) -> str:
        return json.dumps(
            {
                "meta": self.meta,
                "vocab": self.vocab,
                "matcher": json.loads(self.matcher.to_json()),
                "contract": self.contract.to_dict(),
                "special_tokens": list(self.special_tokens),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def loads(s: str) -> "CITArtifact":
        obj = json.loads(s)
        meta = obj.get("meta") or {"schema_version": CIT_ARTIFACT_SCHEMA_VERSION}
        return CITArtifact(
            meta={str(k): v for k, v in meta.items()},
            vocab={str(k): int(v) for k, v in obj["vocab"].items()},
            matcher=CompiledMatcher.from_json(json.dumps(obj["matcher"], ensure_ascii=False)),
            contract=ContractConfig.from_dict(obj["contract"]),
            special_tokens=[str(x) for x in obj.get("special_tokens", [])],
        )


class CITRuntime:
    """Deterministic runtime for a compiled CIT artifact."""

    def __init__(self, artifact: CITArtifact):
        self.artifact = artifact
        self._contract = Contract(artifact.contract)
        self._unk_id = artifact.vocab.get("[UNK]", 1)
        # optional: allow single-character fallback if in vocab
        self._char_vocab: Dict[str, int] = {c: i for c, i in artifact.vocab.items() if len(c) == 1}

    def apply_contract(self, text: str) -> str:
        return self._contract.apply(text)

    def encode(self, text: str) -> List[int]:
        x = self.apply_contract(text)
        return self.artifact.matcher.encode_greedy(x, unk_id=self._unk_id, vocab_id_for_char=self._char_vocab)

    def decode(self, ids: Sequence[int]) -> str:
        it = self.artifact.id_to_token
        return "".join(it[i] if 0 <= i < len(it) else "" for i in ids)