# Domain-Specific Tokenization (DST)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Reproducible](https://img.shields.io/badge/reproducible-✓-green.svg)]()

> A deterministic, grammar-guided tokenizer system for structured data — 100 % reversible, linear-time, and drop-in compatible with Transformers.

---

## ✨ Overview

**Domain-Specific Tokenization (DST)** provides a *formal, efficient, and invertible* framework for encoding structured data such as HTTP logs, configuration files, source code, or biosequences.

It guarantees:
- ✅ **Perfect round-trip fidelity** – every input string can be exactly reconstructed.
- ⚙️ **Deterministic finite-state encoding** – compiled into DFSTs with $O(|x|)$ complexity.
- 🧩 **Grammar-aware vocabularies** – guided by domain regular expressions and schemas.
- 🤝 **Hugging Face compatibility** – exports `tokenizer.json` for existing Transformer stacks.

---

## 🚀 Quick Start

### 1️⃣ Install
```bash
git clone git@github.com:puyanguvic/Domain-Specific-Tokenization.git
cd Domain-Specific-Tokenization
pip install -e .
````

### 2️⃣ Train a tokenizer

```bash
dst train --input examples/sample_corpus.txt --output tokenizer.json
```

### 3️⃣ Encode / Decode in Python

```python
from dst.tokenizer import DSTTokenizer

corpus = ["GET /index.html HTTP/1.1", "Host: example.com"]
tokenizer = DSTTokenizer.train(corpus, min_freq=1)

tokens = tokenizer.encode("GET /index.html HTTP/1.1")
print(tokens)
# ['GET', ' ', '/', 'index', '.', 'html', ' ', 'HTTP', '/', '1', '.', '1']

print(tokenizer.decode(tokens))
# "GET /index.html HTTP/1.1"

assert tokenizer.verify(corpus)
```

### 4️⃣ Save / Load Tokenizer

```python
from dst.tokenizer import DSTTokenizer

# After training
tokenizer.save_json("tokenizer.json")

# Reconstruct later (builds DFST from saved vocab)
tokenizer2 = DSTTokenizer.load_json("tokenizer.json")

assert tokenizer2.verify(corpus)
```

---

## 📂 Repository Structure

| Path                          | Description                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `dst/vocab.py`                | Grammar-guided vocabulary induction (regex extraction, frequency filtering). |
| `dst/dfst.py`                 | Deterministic finite-state transducer (DFST) encoder–decoder.                |
| `dst/tokenizer.py`            | Main training / encoding / export interface.                                 |
| `dst/cli.py`                  | Command-line interface (`dst train`, `dst encode`).                          |
| `examples/sample_corpus.txt`  | Example HTTP corpus.                                                         |
| `tests/test_reversibility.py` | Unit test ensuring κ(τ(x)) = x for all inputs.                               |

---

## ⚙️ Command-Line Interface

```bash
usage: dst <command> [options]

Commands:
  train     Train a deterministic tokenizer from a text corpus
  encode    Encode text using a trained tokenizer

train options:
  --input PATH            Path to training corpus (one line per sample)
  --output PATH           Output tokenizer JSON (default: dst_tokenizer.json)
  --min-freq INT          Minimum frequency for candidate tokens (default: 5)
  --max-vocab INT         Maximum vocabulary size (default: 32000)

encode options:
  --input PATH            Path to text file to encode (line by line)
  --tokenizer PATH        Tokenizer JSON path (default: dst_tokenizer.json)
  --format {plain,json}   Output format (default: plain)
```

Examples:

```bash
# Train tokenizer (defaults)
dst train --input examples/sample_corpus.txt --output tokenizer.json

# Train with custom vocabulary constraints
dst train --input examples/sample_corpus.txt \
         --output tokenizer.json \
         --min-freq 1 --max-vocab 4096

# Encode a text file (space-separated tokens per line)
dst encode --input examples/sample_corpus.txt --tokenizer tokenizer.json

# Encode as JSON (one JSON array per line)
dst encode --input examples/sample_corpus.txt --tokenizer tokenizer.json --format json
```

Notes:
- `--format plain` prints space-separated tokens per input line.
- `--format json` prints one JSON array per input line (JSON Lines).

---

## 🧪 Testing

Run the unit tests to verify reversibility on the sample corpus:

```bash
python3 -m pytest -q
```

---

## 🧠 Design Highlights

DST models tokenization as paired mappings between strings and token sequences:
[
\tau: \Sigma^* \to \mathcal{V}^*, \quad \kappa: \mathcal{V}^* \to \Sigma^*, \quad \kappa(\tau(x)) = x
]

It ensures:

* **Non-erasingness:** every token emits ≥ 1 symbol.
* **Prefix-freeness:** unique segmentation, no ambiguity.
* **Bounded preimage:** finite inverse mappings ⇒ linear-time DFST.

The compiled automaton performs deterministic, auditable transformations suitable for large-scale enterprise or scientific data processing.

---

## 🧪 Example Performance

| Property           | Value                                      |   |   |
| ------------------ | ------------------------------------------ | - | - |
| Reversibility      | ✅ 100 %                                    |   |   |
| Avg Token Length   | ≈ 4.2 chars                                |   |   |
| Sequence Reduction | 10–20 % vs Byte-BPE                        |   |   |
| Complexity         | O(                                         | x | ) |
| Export             | `tokenizer.json` (Hugging Face-compatible) |   |   |

---

## 📜 License

Released under the **MIT License**
© 2025 Pu Yang

---

## 🔗 Related Work

* Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016
* Xue et al., *ByT5: Towards a Token-Free Future with Byte-Level Models*, TACL 2022
* Ding et al., *Byte-Level Tradeoffs in Tokenization*, NeurIPS 2023

---

## 🧪 Experiments

This repo includes a simple, self-contained experiments runner that follows the evaluation design in the paper:

- Measures average tokens per sequence, compression ratio vs. characters, exact round-trip accuracy, and encoding throughput.
- Supports domain profiles (protocol, config, code, bio) via grammar patterns.
- Includes local baselines without external dependencies: byte-level and whitespace.

Prepare a corpus file with one sample per line (e.g., `examples/sample_corpus.txt`).

Examples:

```bash
# Generic profile on the sample corpus
python3 -m experiments.run_experiments \
  --corpus examples/sample_corpus.txt \
  --domain generic \
  --min-freq 1 --max-vocab 4096 \
  --trials 3 --warmup 1 \
  --output experiments/results_generic.json

# Protocol (HTTP) profile
python3 -m experiments.run_experiments \
  --corpus examples/sample_corpus.txt \
  --domain protocol \
  --min-freq 1 --max-vocab 4096 \
  --trials 3 --warmup 1

# Config profile on a local YAML/JSON lines corpus
python3 -m experiments.run_experiments \
  --corpus /path/to/config.jsonl \
  --domain config \
  --limit 50000 \
  --max-vocab 24000 \
  --trials 5 --warmup 1 \
  --output experiments/results_config.json
```

Notes:
- Input formats: `.txt`/`.log` (one line per sample) or `.jsonl` with a `text`/`content`/`value` field.
- Baselines: byte-level is perfectly invertible; whitespace is not (expected to fail round-trip if tested).
- Throughput numbers are CPU timing of the Python DFST; production systems can export DFST tables to native runtimes for higher throughput.
- Ablations: use `--domain generic` to disable grammar priors; sweep `--max-vocab` to study vocabulary-size effects.
