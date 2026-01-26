from __future__ import annotations
import os
from ...interface.contract import Contract, ContractConfig
from ...io.data import iter_text
from ...artifacts.hf_artifact import save_hf_tokenizer
from ...artifacts.hygiene_artifact import HygieneArtifact, resolve_versions, save_hygiene_artifact
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def train_wordpiece_hygiene(
    corpus: str,
    outdir: str,
    vocab_size: int,
    contract_cfg: ContractConfig,
    fmt: str = "txt",
    text_key: str = "text",
    max_samples: int | None = None,
    min_frequency: int = 10,
    continuing_subword_prefix: str = "##",
    model_max_length: int = 512,
    clean: bool = True,
    hygiene_outdir: str | None = None,
    tokenizer_version: str | None = None,
    hygiene_version: str | None = None,
    version: str | None = None,
    emit_contract_in_tokenizer_dir: bool = False,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    contract = Contract(contract_cfg)

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", continuing_subword_prefix=continuing_subword_prefix))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        continuing_subword_prefix=continuing_subword_prefix,
        special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"],
    )

    def gen():
        for s in iter_text(corpus, fmt=fmt, text_key=text_key, max_samples=max_samples, clean=clean):
            yield contract.apply(s)

    tokenizer.train_from_iterator(gen(), trainer=trainer)

    resolved_versions = None
    if hygiene_outdir is not None or tokenizer_version or hygiene_version or version:
        tokenizer_version, hygiene_version = resolve_versions(
            tokenizer_version=tokenizer_version,
            hygiene_version=hygiene_version,
            version=version,
        )
        resolved_versions = (tokenizer_version, hygiene_version)

    save_hf_tokenizer(
        tokenizer,
        outdir,
        model_max_length=model_max_length,
        extra_config={"continuing_subword_prefix": continuing_subword_prefix},
        tokenizer_version=tokenizer_version,
        hygiene_version=hygiene_version,
    )

    if hygiene_outdir is not None:
        tok_ver, hyg_ver = resolved_versions  # type: ignore[misc]
        save_hygiene_artifact(
            HygieneArtifact(
                hygiene_version=hyg_ver,
                tokenizer_version=tok_ver,
                contract=contract_cfg,
            ),
            hygiene_outdir,
            overwrite=True,
        )

    if emit_contract_in_tokenizer_dir or hygiene_outdir is None:
        with open(os.path.join(outdir, "cit_contract.json"), "w", encoding="utf-8") as f:
            f.write(contract_cfg.to_json())
