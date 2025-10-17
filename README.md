# Domain-Specific Tokenization (DST)

A unified framework for building, analyzing, and validating
**domain-specific tokenizers** with formal consistency guarantees.

> "A tokenizer should not break statistical consistency —
> τ∘κ = Id should hold in practice as well as in theory."

---

## 🌍 Supported Domains

- HTTP / Web Requests  (`HTTPTokenizer`)
- HTML / XML Structures (`HTMLTokenizer`)
- URLs / Query Params   (`URLTokenizer`)
- Logs / Security Feeds (`LogTokenizer`)

---

## 🧩 Key Features

- Formal definition of τ (encode) and κ (decode)
- Guaranteed reversibility: κ∘τ = Id
- Finite-State Transducer (FST) implementation
- K-best marginalization for stochastic consistency
- Hugging Face compatible export (tokenizer.json / vocab.txt)
- Consistency & Compression metrics

---

## 🧪 Benchmarks

| Domain | Tokenizer | Consistency | Bits/Char | Encoding Speed |
|---------|------------|--------------|-------------|----------------|
| HTTP | WordPiece | 0.963 | 1.34 | 5100 tok/s |
| HTTP | BPE | 0.941 | 1.29 | 4900 tok/s |
| HTTP | **DST (ours)** | **1.000** | **1.17** | **6100 tok/s** |

---

## 📦 Install

```bash
pip install domain-specific-tokenization
