from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

from ..interface.contract import ContractConfig

HYGIENE_ARTIFACT_SCHEMA_VERSION = "hygiene_artifact.v1"


@dataclass(frozen=True)
class HygieneArtifact:
    schema_version: str = HYGIENE_ARTIFACT_SCHEMA_VERSION
    hygiene_version: str = "unversioned"
    tokenizer_version: str = "unversioned"
    contract: ContractConfig = ContractConfig()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hygiene_version": self.hygiene_version,
            "tokenizer_version": self.tokenizer_version,
            "contract": self.contract.to_dict(),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def loads(s: str) -> "HygieneArtifact":
        data = json.loads(s)
        return HygieneArtifact(
            schema_version=data.get("schema_version", HYGIENE_ARTIFACT_SCHEMA_VERSION),
            hygiene_version=str(data.get("hygiene_version", "unversioned")),
            tokenizer_version=str(data.get("tokenizer_version", "unversioned")),
            contract=ContractConfig.from_dict(data.get("contract", {})),
        )


def resolve_versions(
    *,
    tokenizer_version: Optional[str],
    hygiene_version: Optional[str],
    version: Optional[str] = None,
) -> Tuple[str, str]:
    if version:
        if tokenizer_version is None:
            tokenizer_version = version
        if hygiene_version is None:
            hygiene_version = version
    if not tokenizer_version or not hygiene_version:
        raise ValueError("tokenizer_version and hygiene_version are required for hygiene artifacts")
    return str(tokenizer_version), str(hygiene_version)


def save_hygiene_artifact(
    artifact: HygieneArtifact,
    outdir: str,
    *,
    overwrite: bool = False,
) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hygiene_artifact.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"hygiene_artifact.json already exists: {path}")
    path.write_text(artifact.dumps(), encoding="utf-8")
    return path


def load_hygiene_artifact(path: str | Path) -> HygieneArtifact:
    p = Path(path)
    if p.is_dir():
        p = p / "hygiene_artifact.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing hygiene_artifact.json at: {p}")
    return HygieneArtifact.loads(p.read_text(encoding="utf-8"))
