"""Demo: train a ctoken tokenizer from a WAF parquet file.

Usage example:

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

Then validate:

python demo_train_waf.py
"""

from transformers import AutoTokenizer

TOK_DIR = "./ctoken_waf_unigram_2048"

tok = AutoTokenizer.from_pretrained(TOK_DIR)
print(type(tok), type(tok.backend_tokenizer))
print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("GET /a.php?id=1&x=../etc/passwd HTTP/1.1"))
print(tok.tokenize("<METHOD> GET\n<URL> /a.php?id=1\n<PROT> HTTP/1.1\n<HDR>\nHost: x\n<BODY>\n\n")[:60])
