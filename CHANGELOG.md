## 0.7.0
- Added `cit export-hf` (pure-data HF-style export, no remote code).
- Added `cit dataset pull` helper for downloading HF datasets and saving to parquet (`.[hf]` extra).
- Bumped SDK version to 0.7.0.

## 0.6.0
- Unified build config schema and single-entry CLI (`cit`) with subcommands.

## 0.5.0
- Added artifact schema metadata, validation utilities, and typed-symbol integrity tests.
- Added structured logging helpers and CLI log-level flag.

# Changelog

## 0.3.0
- Added **parallel ablation baselines**: BPE+Hygiene, WordPiece+Hygiene, Unigram+Hygiene.
- Documented baseline reproducibility notes in `docs/REPRO_BASELINES.md`.
- Updated README with quick commands and AutoTokenizer loading example.
