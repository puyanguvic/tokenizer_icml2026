# API

The public runtime interface lives in `ctok_core.tokenization.runtime.TokenizerRuntime`.
For model integration use `ctok_core.tokenization.tokenizer.CtokTokenizer`, which provides
a Transformers-style `__call__` returning `input_ids` and `attention_mask`. If you are
using Hugging Face `transformers`, use `ctok_core.tokenization.hf.CtokHFTokenizer`.

Dataset loading uses `ctok_extras.datasets.load_dataset` with JSONL/TSV/text configs or
`hf_dataset` entries.

The build CLI supports optional label-aligned distortion proxies (`--labels`) and
boundary-aware candidate generation (`--boundary-aware`).
Set `base_charset: byte` in tokenizer config to include all 256 byte symbols.

Fine-tuning helpers live in `ctok_extras.models.train.train_roberta` and are surfaced via
`scripts/finetune_roberta.py`.
