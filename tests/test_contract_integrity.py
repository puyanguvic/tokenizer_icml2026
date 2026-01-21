from __future__ import annotations

from cit_tokenizers.interface.contract import Contract, ContractConfig
from cit_tokenizers.cit.compiler import compile_trie
from cit_tokenizers.cit.runtime import CITArtifact, CITRuntime
from cit_tokenizers.cit.validate import validate_artifact


def test_typed_hygiene_replaces_ipv4_and_runtime_emits_atomically() -> None:
    cfg = ContractConfig(enable_json_serialization=False, enable_typed_hygiene=True)
    contract = Contract(cfg)

    raw = "SRC=192.168.1.10 DST=10.0.0.5 PORT:443"
    hygiened = contract.apply(raw)
    assert "<IPV4>" in hygiened, hygiened
    assert "<PORT>" in hygiened, hygiened

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "SRC": 5,
        "DST": 6,
        "=": 7,
        "<IPV4>": 8,
        "<PORT>": 9,
        " ": 10,
    }
    matcher = compile_trie([(s, i) for s, i in vocab.items() if not s.startswith("[")])
    art = CITArtifact(
        meta={"schema_version": "cit_artifact.v1", "builder": "unit_test"},
        vocab=vocab,
        matcher=matcher,
        contract=cfg,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    rt = CITRuntime(art)
    ids = rt.encode(raw)
    toks = [art.id_to_token[i] for i in ids]
    assert "<IPV4>" in toks
    assert "<PORT>" in toks

    issues = validate_artifact(art, typed_symbols=["<IPV4>", "<PORT>"])
    assert issues == [], [i.message for i in issues]
