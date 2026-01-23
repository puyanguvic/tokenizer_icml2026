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

Global:

- `--log-level`: Logging level (e.g., `INFO`, `DEBUG`).

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
