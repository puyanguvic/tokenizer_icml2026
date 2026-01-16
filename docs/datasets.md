# Datasets

This project assumes JSONL/TSV inputs or Hugging Face datasets with explicit text and
label fields so the same loader can cover structured requests, HTML documents, and log
sequences.

## WAF/HTTP request attacks

- Task: request-level classification (benign vs attack).
- Hugging Face dataset: `puyang2025/waf_data_v2`.
- Expected fields: `method`, `url`, `protocol`, `headers`, `body`, `label` (combined into a single request text).
- Labels: `normal`, `anomalous` mapped to 0/1.
- Notes: keep delimiters and field order; do not normalize away boundary tokens.

## Phishing HTML

- Task: HTML phishing detection (phish vs benign).
- Hugging Face dataset: `puyang2025/phish_html`.
- Expected fields: `text` (raw HTML), `label`.
- Labels: `benign`, `malicious` mapped to 0/1.
- Notes: keep tag/attribute boundaries; minimal normalization only.

## HDFS v1 (LogHub)

- Task: sequence-level anomaly detection (normal vs abnormal).
- Hugging Face dataset: `honicky/hdfs-logs-encoded-blocks`.
- Expected fields: `event_encoded` (encoded log sequence), `label`.
- Optional fields: `block_id` (session identifier), `tokenized_block` (pre-tokenized sequence).
- Labels: `Normal`, `Anomaly` mapped to 0/1.
- Notes: sequences are joined with `sequence_delimiter` in config.

## Formats

- Hugging Face datasets: set `hf_dataset` and optional `hf_split`/`hf_name`.
- For streaming datasets set `hf_streaming: true` and `max_samples`.
- JSONL: one JSON object per line with text and label fields.
- TSV/CSV: use `columns` in the config or set `text_index`/`label_index`.
- Text: raw text only (no labels), used for tokenizer builds.
