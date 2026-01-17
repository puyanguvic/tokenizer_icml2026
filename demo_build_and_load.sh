#!/usr/bin/env bash
set -e

# Demo: build from a tiny corpus and load via AutoTokenizer

DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$DIR/_demo_artifact"
CORPUS="$DIR/_demo_corpus.txt"

cat > "$CORPUS" <<'EOF'
GET /index.html?x=1&y=2
POST /login?user=admin&pass=123
GET /search?q=test&lang=en
EOF

python "$DIR/build_ctok_from_corpus.py" \
  --corpus "$CORPUS" \
  --format txt \
  --outdir "$WORK" \
  --vocab_size 2048 \
  --max_len 10 \
  --min_freq 1 \
  --max_samples 100000 \
  --emit_code

python - <<'PY'
from transformers import AutoTokenizer

# Load from local directory
# NOTE: trust_remote_code=True is required for custom tokenization_ctok.py

tok = AutoTokenizer.from_pretrained("./_demo_artifact", trust_remote_code=True)
print("Tokenizer class:", type(tok))
print(tok.tokenize("GET /index.html?x=1&y=2"))
enc = tok("GET /index.html?x=1&y=2", padding="max_length", truncation=True, max_length=32)
print({k: v[:16] for k, v in enc.items()})
PY

echo "OK"
