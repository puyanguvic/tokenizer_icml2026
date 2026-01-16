# Reproducibility

## Data and artifacts

- Record dataset hashes, schema versions, and normalization configs.
- Store tokenizer artifacts with `vocab.json`, `rules.json`, and `manifest.json` hashes.

## Training protocols

- Equal-model: fix architecture, optimizer, batch size, max length, and steps.
- Equal-compute: fix total tokens processed (optionally wall-clock budget).

## Metrics to log

- Token length: mean, p95, p99, max, token-to-byte ratio.
- Throughput: tokens/sec, samples/sec, step time, peak memory.
- Detection: AUROC, F1, accuracy (task dependent).

## Seeds and splits

- Log random seeds, split definitions, and probe refresh schedules.
- Keep golden tokenization cases for determinism checks.
