from __future__ import annotations

from pathlib import Path


def test_cit_artifact_roundtrip_and_encode(tmp_path: Path) -> None:
    """Fast, dependency-light unit test.

    We avoid importing `transformers` in unit tests to keep CI lightweight and
    to ensure the core artifact + runtime are stable independently.
    """
    from cit_tokenizers.cit.compiler import compile_trie
    from cit_tokenizers.cit.runtime import CITArtifact, CITRuntime
    from cit_tokenizers.interface.contract import ContractConfig

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "GET": 5,
        "/": 6,
        "index": 7,
    }

    matcher = compile_trie([(s, i) for s, i in vocab.items() if not s.startswith("[")])
    artifact = CITArtifact(
        meta={"schema_version": "cit_artifact.v1", "builder": "unit_test"},
        vocab=vocab,
        matcher=matcher,
        contract=ContractConfig(enable_json_serialization=False, enable_typed_hygiene=False),
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # JSON round-trip
    s = artifact.dumps()
    artifact2 = CITArtifact.loads(s)
    assert artifact2.vocab == artifact.vocab
    assert artifact2.matcher.max_token_len == artifact.matcher.max_token_len

    # Runtime encode should be deterministic
    rt = CITRuntime(artifact2)
    assert rt.encode("GET/index") == [5, 6, 7]
