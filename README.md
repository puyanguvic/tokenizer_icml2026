# ctoken (merged trainer + artifact builder)

This package standardizes CTok training inputs to **Parquet** and provides a **pluggable preprocessing layer** that maps dataset-specific schemas (e.g., WAF HTTP fields) into a single `text` column suitable for tokenizer training. It also ships the **Unigram artifact builder** so no separate `ctok_core` directory is needed.

## What you get
- Parquet-only ingestion (`ctoken.io.parquet`)
- Preprocess registry (`ctoken.preprocess.registry`) + WAF schema preprocessor (`waf_http_v2`)
- Built-in Unigram artifact builder that outputs a standard HF `tokenizer.json` (no `trust_remote_code`)
- A CLI that:
  1) Converts raw parquet -> aligned parquet with `text` (and `label`)
  2) Trains and exports a tokenizer artifact

## Install

From the directory containing this repo:

```bash
pip install -e .
```

## Demo (WAF dataset)

1) Download `train.parquet` from HF (or use a local parquet file with columns: `method,url,protocol,headers,body,label`).
2) Run:

```bash
python -m ctoken.cli \
  --src_parquet /path/to/train.parquet \
  --preprocess waf_http_v2 \
  --work_parquet ./_work/aligned_waf_train.parquet \
  --outdir ./ctoken_waf_unigram_2048 \
  --vocab_size 2048 \
  --model_max_length 512 \
  --max_len 16 \
  --sample_words 200000 \
  --prune_iters 4 \
  --prune_frac 0.20 \
  --num_workers 0
```

Then validate:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./ctoken_waf_unigram_2048")
print(type(tok.backend_tokenizer))
print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("GET /a.php?id=1&x=../etc/passwd HTTP/1.1"))
```

## Add a new dataset schema
Create `ctoken/preprocess/<name>.py` and register a preprocessor with `@register(PreprocessSpec(...))`.
