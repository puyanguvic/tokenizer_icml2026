# CIT Tokenizers

This package provides:

- **CIT**: Controlled Interface Tokenization (contract-constrained tokenizer interface).
- **Parallel, hygiene-aligned baselines**:
  - `BPE+Hygiene` (via `cit train bpeh`)
  - `WordPiece+Hygiene` (via `cit train wordpieceh`)
  - `Unigram+Hygiene` (via `cit train unigramh`)

All methods share the **same deterministic interface contract** (JSON role-aware serialization + typed hygiene + integrity rules),
and differ only in **vocabulary induction / segmentation model**.

## Install (editable)

```bash
pip install -e .
```

## Quickstart

1) Train a CIT tokenizer:

```bash
cit train cit --corpus path/to/corpus.txt --outdir tokenizers/cit_demo --vocab-size 8192 --preset waf
```

2) Load with Transformers:

```python
from cit_tokenizers import CITTokenizer
tok = CITTokenizer.from_pretrained("tokenizers/cit_demo")
print(tok.tokenize("GET /index.html?x=1 HTTP/1.1"))
```

Note: `CITTokenizer` loads data-only artifacts without `trust_remote_code`; `AutoTokenizer.from_pretrained(...)` is not supported.

## Train a CIT tokenizer

Example (WAF corpus):

```bash
cit train cit \
  --preset waf \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --outdir tokenizers/waf_cit_tokenizer \
  --vocab-size 8192 \
  --min-freq 10 \
  --seed 0
```

Training cleans noisy blobs (e.g., base64/hex) by default; disable with `--no-clean`.
For `--preset http|waf`, structured HTTP parsing is enabled by default (see `--structured-input`).

Explicitly enable structured parsing:

```bash
cit train cit \
  --preset waf \
  --structured-input http \
  --structured-max-len 4096 \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --outdir tokenizers/waf_cit_tokenizer_struct \
  --vocab-size 8192 \
  --min-freq 10 \
  --seed 0
```

Structured HTTP preprocessing:
- Emits tags like `<METHOD>`, `<URL>`, `<HDR>`, `<BODY>` and splits query/cookie/body values into key/value tokens.
- High-entropy values are replaced with placeholders such as `<HEX>`, `<B64>`, `<HASH_32>`, `<UUID>`, `<URLENC>`, `<BYTES>`, and numeric buckets.
- Cookie keys keep readable prefixes while high-entropy suffixes are normalized (e.g., `comment_author_<hash>` -> `comment_author_<HEX>`).
- Fallback on parse failure: `<RAW> ... <TRUNC>` with a hard length cap.

Optional: pre-clean a corpus and write it back out:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --out datasets/corpus/puyang2025__waf_data_v2_train_clean.jsonl
```

Optional: pre-clean + structured HTTP parsing:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --structured-input http \
  --out datasets/corpus/puyang2025__waf_data_v2_train_struct.jsonl
```

Cookie key normalization demo:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --structured-input http \
  --out datasets/corpus/puyang2025__waf_data_v2_train_struct_cookiekeys.jsonl
```

## Verify a trained tokenizer (sample 5)

After training, load the artifact and tokenize a few samples from the corpus (match `fmt`/`text_key` and `clean` to your training flags):

```bash
python - <<'PY'
from cit_tokenizers import CITTokenizer
from cit_tokenizers.io.data import iter_text

tok = CITTokenizer.from_pretrained("tokenizers/waf_cit_tokenizer_struct")

for i, text in enumerate(
    iter_text(
        "datasets/corpus/puyang2025__waf_data_v2_train.jsonl",
        fmt="jsonl",
        text_key="text",
        max_samples=5,
        clean=True,
    ),
    1,
):
    print(f"--- sample {i} ---")
    print(text)
    print(tok.tokenize(text))
PY
```

Common arguments:

- `--corpus`: Path to corpus file (required).
- `--format`: Corpus format: `txt`, `jsonl`, or `parquet` (default: `txt`).
- `--text-key`: Text field for `jsonl`/`parquet` (default: `text`).
- `--max-samples`: Optional cap on samples read from the corpus.
- `--outdir`: Output artifact directory (required).
- `--vocab-size`: Target vocabulary size (default: `8192`).
- `--min-freq`: Minimum token frequency (default: `10`).
- `--preset`: Domain preset: `default`, `http`, or `waf` (default: `default`).
- `--seed`: RNG seed (default: `0`).
- `--lambda-rd`: Regularization strength for candidate scoring (default: `0.0`).
- `--distortion-mode`: Distortion proxy: `none` or `boundary_penalty` (default: `none`).
- `--boundary-penalty`: Penalty scalar for `boundary_penalty` mode (default: `1.0`).
- `--config`: Path to a `CITBuildConfig` JSON file (overrides per-flag trainer/contract).

Contract options:

- `--contract-json`: Path to a `ContractConfig` JSON file.
- `--no-json-serialization`: Disable JSON role-aware serialization.
- `--no-typed-hygiene`: Disable typed hygiene.
- `--no-numeric-buckets`: Disable numeric bucket tokens.
- `--long-num-min-digits`: Minimum digit count for long-number buckets (default: `6`).
- `--structured-input`: Structured preprocessing mode: `none`, `http`, or `waf` (default: auto for `http|waf` preset).
- `--structured-max-len`: Hard cap for structured parsing input length (default: `4096`).

Global:

- `--log-level`: Logging level (e.g., `INFO`, `DEBUG`).

## External Hygiene + HF fast tokenizer (production)

Online path:

```
raw HTTP -> hygiene.normalize() -> hf_tokenizer.encode()
```

Artifacts:

- **Hygiene artifact**: `hygiene_artifact.json` (ContractConfig + versions).
- **HF tokenizer artifact**: standard HF files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).

HF tokenizer contains:

- Special tokens (e.g., `<METHOD>`, `<URL_RAW>`, `<URL_DEC1>`, `<HAS_CRLF>`, `<HEX>`).
- Model params + vocab for BPE/WordPiece/Unigram.
- Post-processor (e.g., CLS/SEP) in `tokenizer.json`.

Train a baseline with external hygiene:

```bash
cit train wordpieceh \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --outdir tokenizers/waf_wordpieceh_v1 \
  --hygiene-outdir hygiene/waf_hyg_v1 \
  --version v1 \
  --vocab-size 8192 \
  --min-freq 10
```

Runtime usage:

```python
from cit_tokenizers.hygiene import load_hygiene_runtime, load_tokenizer_config, assert_version_binding
from transformers import AutoTokenizer

hyg = load_hygiene_runtime("hygiene/waf_hyg_v1")
tok = AutoTokenizer.from_pretrained("tokenizers/waf_wordpieceh_v1", use_fast=True)
assert_version_binding(hyg.artifact, load_tokenizer_config("tokenizers/waf_wordpieceh_v1"))

ids = tok.encode(hyg.normalize(raw_http))
```

Notes:
- When `--hygiene-outdir` is set, `--tokenizer-version`/`--hygiene-version` (or `--version`) are required.
- Use `--emit-contract` if you also want `cit_contract.json` in the tokenizer dir for legacy tooling.

## Package layout (v0.7.0)

The library is structured as a small, production-oriented SDK:

- `cit_tokenizers.interface`: contract, value-aware hygiene, serialization
- `cit_tokenizers.cit`: CIT trainer, compiler, runtime matcher
- `cit_tokenizers.artifacts`: artifact IO helpers
- `cit_tokenizers.io`: corpus/dataset IO

## CLI recipes

### Export a CIT artifact as a HuggingFace-style folder (no remote code)

```bash
cit export-hf \
  --artifact-dir tokenizers/cit_demo \
  --outdir tokenizers/cit_demo_hf \
  --model-max-length 512
```

Load:

```python
from cit_tokenizers import CITTokenizer
tok = CITTokenizer.from_pretrained("tokenizers/cit_demo_hf")
```

### Pull a HuggingFace dataset split to parquet

Requires optional dependencies:

```bash
pip install -e '.[hf]'
```

Then:

```bash
cit dataset pull \
  --dataset imdb \
  --split train \
  --text-key text \
  --label-key label \
  --out data/imdb_train.parquet \
  --max-samples 10000 \
  --shuffle
```

## Development

Install (editable) with dev tooling:

```bash
pip install -e '.[dev]'
pre-commit install
```

Run checks locally:

```bash
ruff check .
ruff format .
pytest
```
