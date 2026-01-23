from __future__ import annotations

from cit_tokenizers.interface.contract import Contract, ContractConfig
from cit_tokenizers.cit.validate import validate_artifact
from cit_tokenizers.cit.compiler import compile_trie
from cit_tokenizers.cit.runtime import CITArtifact


def test_hex_blob_replaced_even_when_adjacent_to_alnum() -> None:
    cfg = ContractConfig(enable_json_serialization=False, enable_typed_hygiene=True, typed_hygiene_mode="http")
    contract = Contract(cfg)

    raw = "Cookie: session=abc1aefbe2f76edd740f8e362f39da3353bdef; Path=/"
    hygiened = contract.apply(raw)
    assert "<HEX>" in hygiened or "<HASH>" in hygiened, hygiened


def test_validate_flags_many_long_hex_tokens_when_typed_symbols_exist() -> None:
    # Minimal matcher over a tiny vocab (not used by validate).
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "<HEX>": 5,
        "GET": 6,
        " ": 7,
    }
    # Inject high-entropy contamination (should be caught).
    for i in range(50):
        vocab[f"{i:02x}" * 16] = len(vocab)  # 32 hex chars

    matcher = compile_trie([("GET", 6), (" ", 7)])
    art = CITArtifact(
        meta={"schema_version": "cit_artifact.v1"},
        vocab=vocab,
        matcher=matcher,
        contract=ContractConfig(enable_json_serialization=False, enable_typed_hygiene=True, typed_hygiene_mode="http"),
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    typed_symbols = Contract(art.contract).typed_symbols()
    issues = validate_artifact(art, typed_symbols, max_long_hex_count=10)  # make it strict for the test
    assert any(iss.code in ("entropy_hex_overflow", "typed_hygiene_leak_hex") for iss in issues), issues
