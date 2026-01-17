# CTok Tokenizer (prototype + AutoTokenizer-loadable)

This bundle includes:

- `tokenization_ctok.py`: a minimal **CTokTokenizer** (runtime) implementing **deterministic left-to-right longest match**.
- `build_ctok_from_corpus.py`: a build-time script that generates a loadable tokenizer artifact directory with:
  - `vocab.json`
  - `ctok_meta.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - (optionally) copies `tokenization_ctok.py` into the artifact

## Quick demo

```bash
./demo_build_and_load.sh
```

## Build a tokenizer artifact from your corpus

Assume your corpus is one sample per line:

```bash
python build_ctok_from_corpus.py \
  --corpus data/train.txt \
  --format txt \
  --outdir ctok_artifact \
  --vocab_size 8192 \
  --max_len 12 \
  --min_freq 50 \
  --boundaries "=&?:/\n\t <>\"'<>" \
  --emit_code
```

Then load:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./ctok_artifact", trust_remote_code=True)
print(tok.tokenize("GET /index.html?x=1&y=2"))
```

## Parquet corpora

If your corpus is stored in **Parquet** (single file or a directory of parquet shards), use:

```bash
python build_ctok_from_corpus.py \
  --corpus data/train.parquet \
  --format parquet \
  --text_key text \
  --label_key label \
  --outdir ctok_artifact \
  --vocab_size 8192 \
  --max_len 12 \
  --min_freq 50 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --emit_code
```

Notes:
- Parquet streaming reads use `pyarrow.dataset` when available (recommended). If `pyarrow` is not installed, the script falls back to `pandas.read_parquet` (loads selected columns into memory).
- `--max_samples` limits how many rows are processed (useful for fast sweeps).

## Optional: label-aware semantic scoring (fast proxy)

If your corpus is labeled (binary or multi-class), you can provide it as TSV:

```
<label>\t<text>
```

Then enable a lightweight semantic score using mutual information (MI) between labels and token presence:

```bash
python build_ctok_from_corpus.py \
  --corpus data/train.tsv \
  --format tsv \
  --outdir ctok_artifact \
  --vocab_size 8192 \
  --max_len 12 \
  --min_freq 50 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --semantic_top_k 50000 \
  --emit_code
```

This keeps the runtime tokenizer unchanged, but biases the vocabulary toward tokens that are both
compressive and label-informative.

## Notes

- Compression-only build corresponds to `lambda_sem=0`.
- The MI proxy is a **fast stand-in** for the paper's probe-based directed distortion control; you can later swap it
  with a real probe estimate of `delta(c)` without changing the runtime code.
- Runtime is Python; you can later replace the matcher with a Rust extension without changing the external API.
