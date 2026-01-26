from __future__ import annotations


def _make_trainer():
    from cit_tokenizers.cit.trainer import CITTrainer, CITTrainerConfig
    from cit_tokenizers.config import CITBuildConfig
    from cit_tokenizers.interface.contract import ContractConfig

    cfg = CITTrainerConfig(
        vocab_size=256,
        min_freq=1,
        len_max=12,
        preset="http",
        sample_texts=None,
    )
    contract_cfg = ContractConfig(
        enable_json_serialization=False,
        enable_typed_hygiene=False,
        structured_input_mode="http",
    )
    build_cfg = CITBuildConfig(trainer=cfg, contract=contract_cfg)
    return CITTrainer(build_config=build_cfg)


def test_http_struct_tokens_reserved(tmp_path) -> None:
    from cit_tokenizers.interface.http_struct import HTTP_STRUCT_TOKENS

    trainer = _make_trainer()
    sample = "<METHOD> GET\n<URL> /a\n<PROT> HTTP/1.1\n<HDR>\nHost: example.com\n"
    art = trainer.train_from_iterator([sample], tmp_path)

    for tok in HTTP_STRUCT_TOKENS:
        assert tok in art.vocab


def test_http_tag_internals_skipped() -> None:
    trainer = _make_trainer()
    sample = "<METHOD> GET\n<URL> /a\n<PROT> HTTP/1.1\n<HDR>\nHost: example.com\n"
    processed = trainer._contract.apply(sample)
    cand = trainer._extract_candidates([processed])

    for tag in ("METHOD", "URL", "PROT", "HDR"):
        assert tag not in cand
