# Controlled Tokenization Core (ctok-core)

Controlled Tokenization (ctok-core) is a deterministic, grammar-guided tokenizer system for
structured and semi-structured detection workloads. The repository provides a clean
runtime tokenizer, a configurable vocabulary induction pipeline, and scaffolding for
experiments and diagnostics aligned with the paper.

## Quickstart

### Prerequisites

- Python 3.10+ available on PATH.
- `uv` installed (see https://github.com/astral-sh/uv).

### Setup

```bash
./setup.sh
source .venv/bin/activate
source .ctok_env
```

`setup.sh` writes `.ctok_env` with default task/model/tokenizer values for the training CLI.

If you want a different venv path or extras:

```bash
VENV_DIR=/tmp/ctok-venv ./setup.sh
./setup.sh --extras hf,transformers
./setup.sh --no-extras
./setup.sh --no-dev
```

### Build and run

```bash
python -m ctok_extras.cli.main build --corpus data/raw/example.txt --output artifacts/tokenizers/ctok_v1 \
  --config configs/tokenizers/ctok.yaml
python -m ctok_extras.cli.main encode --artifact artifacts/tokenizers/ctok_v1 \
  --input data/raw/example.txt --format ids
python -m ctok_extras.cli.main eval --artifact artifacts/tokenizers/ctok_v1 \
  --corpus data/raw/example.txt
```

Optional: add a label-aligned file to enable the label-entropy distortion proxy:

```bash
python -m ctok_extras.cli.main build --corpus data/raw/example.txt --labels data/raw/example.labels \
  --output artifacts/tokenizers/ctok_v1 --config configs/tokenizers/ctok.yaml
```

For boundary-respecting candidates (boundary-healing baseline), add:

```bash
python -m ctok_extras.cli.main build --corpus data/raw/example.txt --output artifacts/tokenizers/ctok_v1 \
  --config configs/tokenizers/ctok.yaml --boundary-aware
```

### Python usage

```python
from ctok_core.tokenization import CtokTokenizer

tokenizer = CtokTokenizer.from_pretrained("artifacts/tokenizers/ctok_v1")
batch = tokenizer(["GET /", "POST /admin"], padding=True, truncation=True, max_length=128)
print(batch["input_ids"])
```

### Transformers integration

Install the optional dependency:

```bash
uv sync --extra transformers
```

```python
from ctok_core.tokenization.hf import CtokHFTokenizer

hf_tokenizer = CtokHFTokenizer.from_pretrained("artifacts/tokenizers/ctok_v1")
```

### Fine-tuning RoBERTa

```bash
uv sync --extra transformers --extra hf
python -m ctok_extras.cli.main train --task waf_http --model roberta_base --tokenizer ctok_v1 \
  --output results/runs/roberta_waf_http
```

## Layout

- `src/ctok_core/`: core package (tokenization runtime, compiler, induction).
- `ctok_extras/`: datasets, experiments, diagnostics, probes, models, and CLI tooling.
- `configs/`: configuration files for datasets, tokenizers, models, and experiments.
- `scripts/`: one-off helper scripts (data prep, export, end-to-end runs).
- `docs/`: project documentation and paper-facing notes.
- `artifacts/`, `results/`, `data/`: generated outputs (gitignored except README files).

## Notes

The current implementation focuses on deterministic runtime tokenization and a
reference induction loop that follows the gain--distortion structure in the paper.
Specialized probes, diagnostics, and model wrappers are stubbed for extension.
See `docs/experiments.md` for the experiment plan and config templates.
Hugging Face datasets are configured via `configs/datasets/*.yaml` using `hf_dataset`.
Model templates include DistilRoBERTa, RoBERTa-base, and RoBERTa-large configs.
Experiments are intended to run a dataset x model-size matrix across these three sizes.
