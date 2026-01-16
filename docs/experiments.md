# Experiments

This page encodes the experimental setup used in the paper for structured detection.

## Datasets and tasks

- WAF/HTTP attacks: request-level classification (benign vs attack).
- Phishing HTML: document-level classification (phish vs benign).
- HDFS v1 (LogHub): sequence anomaly detection (normal vs abnormal).
  - HF sources: `puyang2025/waf_data_v2`, `puyang2025/phish_html`, `honicky/hdfs-logs-encoded-blocks`.

## Models

- Small: DistilRoBERTa for low-latency settings.
- Base: RoBERTa-base as the main backbone.
- Large: RoBERTa-large to test scale effects.
  - Matrix: run each dataset with all three sizes (small/base/large).
- Tokenizer-free baseline: CANINE (character-level).
- Traditional baseline: character n-gram + logistic regression.

## Tokenizers

All tokenizers share a fixed vocab budget (8k/16k/32k).

- BPE (SentencePiece/BPE).
- Unigram (SentencePiece/Unigram).
- Byte-level (pure byte or byte-BPE).
- Controlled tokenizer (ctok): gain-distortion with deterministic runtime.
- Boundary-healing baseline: boundary-aware runtime that prevents tokens from crossing delimiters.

## Training protocols

- Equal-model: fix architecture, optimizer, batch size, steps, max length.
- Equal-compute: fix total tokens processed (optionally wall-clock).

## Metrics (three-layer)

- Tokenization microbench: throughput, latency percentiles, artifact sizes.
- End-to-end: tokens/sec, samples/sec, step time, peak memory, inference latency.
- Diagnostics: distortion (probe CE/KL), Granger/CMI deltas, geometry metrics,
  boundary sensitivity and stability.

## Experiment blocks

E1 Rate-distortion curves: rate vs distortion across vocab sizes.
E2 Equal-compute: AUROC/F1 under fixed token budget.
E3 Equal-model: system throughput and memory gains.
E4 Boundary sensitivity: segmentation jitter and stability under edits.
E5 Mechanism alignment: distortion vs GC and separability trends.
E6 Ablations: lambda=0, probe capacity, teacher type, boundary-aware candidates,
   runtime variants (dfa/fst/heuristic).
