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

2) Load with Hugging Face:

```python
from cit_tokenizers.tokenization_cit import CITTokenizer
tok = CITTokenizer.from_pretrained("tokenizers/cit_demo")
print(tok.tokenize("GET /index.html?x=1 HTTP/1.1"))
```

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
from cit_tokenizers.tokenization_cit import CITTokenizer
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
