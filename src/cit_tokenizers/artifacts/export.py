from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..cit.runtime import CITArtifact


def export_cit_as_hf_dir(
    artifact_dir: str,
    outdir: str,
    *,
    model_max_length: int = 512,
    overwrite: bool = False,
    extra_tokenizer_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Export a CIT artifact directory into a HuggingFace-style tokenizer folder.

    This does **not** require (and does not embed) any executable code. The output
    directory can be loaded by:

        from cit_tokenizers.tokenization_cit import CITTokenizer
        tok = CITTokenizer.from_pretrained(outdir)

    Notes:
      * `AutoTokenizer.from_pretrained(outdir)` will generally not work because
        CITTokenizer is not part of `transformers` built-ins and we intentionally
        avoid `auto_map` / remote-code execution.
    """

    src = Path(artifact_dir)
    dst = Path(outdir)

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Load CIT artifact
    art_path = src / "cit_artifact.json"
    if not art_path.exists():
        raise FileNotFoundError(f"Missing cit_artifact.json in: {src}")
    artifact = CITArtifact.loads(art_path.read_text(encoding="utf-8"))

    # Copy core files (pure data)
    shutil.copy2(art_path, dst / "cit_artifact.json")
    st_map_src = src / "special_tokens_map.json"
    if st_map_src.exists():
        shutil.copy2(st_map_src, dst / "special_tokens_map.json")
    else:
        # Minimal defaults
        st_map = {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        (dst / "special_tokens_map.json").write_text(
            json.dumps(st_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Provide a conventional vocab.json for tooling convenience
    (dst / "vocab.json").write_text(
        json.dumps(artifact.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Standard tokenizer_config.json (no auto_map)
    tok_cfg: Dict[str, Any] = {
        "tokenizer_class": "CITTokenizer",
        "model_max_length": int(model_max_length),
        "padding_side": "right",
        "truncation_side": "right",
    }
    if extra_tokenizer_config:
        tok_cfg.update(extra_tokenizer_config)
    (dst / "tokenizer_config.json").write_text(
        json.dumps(tok_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Extra metadata for audit/repro
    (dst / "cit_meta.json").write_text(
        json.dumps(artifact.meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
