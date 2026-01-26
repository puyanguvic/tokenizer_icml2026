from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from ..interface.contract import Contract
from ..artifacts.hygiene_artifact import HygieneArtifact, load_hygiene_artifact


class HygieneRuntime:
    def __init__(self, artifact: HygieneArtifact):
        self.artifact = artifact
        self._contract = Contract(artifact.contract)

    def normalize(self, x: Union[str, Dict[str, Any]]) -> str:
        return self._contract.apply(x)


def load_tokenizer_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer_config.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json at: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def assert_version_binding(hygiene: HygieneArtifact, tokenizer_config: Dict[str, Any]) -> None:
    tok_ver = tokenizer_config.get("tokenizer_version")
    hyg_ver = tokenizer_config.get("hygiene_version")
    if tok_ver != hygiene.tokenizer_version or hyg_ver != hygiene.hygiene_version:
        raise ValueError(
            "Hygiene/tokenizer version mismatch: "
            f"hygiene={hygiene.hygiene_version!r}/{hygiene.tokenizer_version!r}, "
            f"tokenizer_config={hyg_ver!r}/{tok_ver!r}"
        )


def load_hygiene_runtime(path: str | Path) -> HygieneRuntime:
    return HygieneRuntime(load_hygiene_artifact(path))
