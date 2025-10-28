# Domain-Specific Tokenization (DST)

Research toolkit for building, exporting, and evaluating tokenizers that stay invertible on structured data.

## Environment

- `uv sync` to resolve the managed virtual environment with `tokenizers`, `datasets`, and dev tooling.
- `uv run python -m experiments.run_experiments --help` for CLI usage.
- `uv run python -m experiments.run_experiments --domain protocol --corpus data/http_logs.jsonl --limit 50000 --output-dir artifacts/protocol` builds the protocol tokenizer, evaluates it, and saves artifacts plus a JSON summary.
- Pass `--dataset bigcode/the-stack:python/train` (or any Hugging Face path) to stream corpora via `datasets`. Use `--text-field` if the payload column is not named `text`.
- Append `--no-baselines` if you want to skip training the BPE / WordPiece / Byte-BPE baselines.

## Repository Map

- `src/dst/` — candidate extraction, scoring, and deterministic tokenizer core.
- `src/http_tokenizer.py` — HTTP-specific normalization and regex grammar prior.
- `src/builder.py` — programmatic API for exporting a tokenizer bundle.
- `src/metrics.py` — round trip, compression, and average token length metrics (script-friendly fallback).
- `experiments/run_experiments.py` — paper-aligned benchmark harness that orchestrates DST + baselines across domains.
- `design/paper.tex` — method and results manuscript containing the theoretical framing used below.

## Experiment Design

The paper evaluates DST on four structured families: protocol traces, configuration files, source code, and biosequences.  
The experiment harness mirrors that setup with domain-specific normalization and grammar priors:

| Domain key | Corpus focus | Normalization hook | Grammar priors (regex sketches) | Suggested sources |
|------------|--------------|--------------------|----------------------------------|-------------------|
| `protocol` | HTTP / network logs | `http_clean_line` (percent-decodes, trims) | URLs, key-value headers, request lines, IPs | LogHub HTTP traces, Suricata / Zeek HTTP datasets |
| `config` | YAML / JSON configs | whitespace + tab normalization | `key:\ value`, list entries, inline JSON blocks | Kubernetes manifests, Terraform modules, Helm charts |
| `code` | Functions & class definitions | trailing-space stripping | `def`, `class`, callsites, imports, string literals | `bigcode/the-stack:python` (HF), GitHub/OpenAI sanitized snippets |
| `bio` | DNA/RNA FASTA | uppercase + whitespace removal | FASTA headers, nucleotide/codon runs | NCBI RefSeq, Ensembl FASTA exports |

### Metrics captured

- `round_trip`: fraction of evaluation samples where decode(encode(x)) matches the normalized input.
- `compression_ratio`: DST token count divided by a regex baseline (`TOKEN_SPLIT_RE`) token count.
- `avg_token_length`: mean character span of domain tokens (longer implies more compression).
- Baselines report the same round-trip and compression metrics for BPE, WordPiece, and Byte-BPE models trained on the normalized corpus.

### Reproducing the paper experiments

1. **Collect corpora** matching the four domains. Convert them to newline-delimited JSON (`{"text": ...}`) or plain text files in `data/`.
   - For Hugging Face datasets, point the CLI at `dataset/config/split` and optionally lower the `--limit` for prototyping.
   - For local crawls (e.g., LogHub HTTP or Helm charts) preprocess with `jq`/`python` to expose a `text` field per record.
2. **Run the harness** for each domain. Example commands:

   ```bash
   uv run python -m experiments.run_experiments \
     --domain protocol \
     --corpus data/http_logs.jsonl \
     --limit 100000 \
     --output-dir artifacts/protocol

   uv run python -m experiments.run_experiments \
     --domain config \
     --dataset bigcode/the-stack:yaml/train \
     --text-field content \
     --limit 80000 \
     --output-dir artifacts/config

   uv run python -m experiments.run_experiments \
     --domain code \
     --dataset bigcode/the-stack:python/train \
     --text-field content \
     --limit 80000 \
     --output-dir artifacts/code

   uv run python -m experiments.run_experiments \
     --domain bio \
     --corpus data/bio.fasta.txt \
     --limit 60000 \
     --output-dir artifacts/bio
   ```

   Each run writes `artifacts/<domain>_results.json` summarizing DST + baseline metrics and saves tokenizer exports (`tokenizer.json`, `vocab.txt`, metadata) when `--output-dir` is provided.

3. **Aggregate results** by stacking the JSON summaries or loading them into a notebook. Compare with Table 1 / Figure 2 in `design/paper.tex`:
   - Expect near-100% round-trip fidelity for DST across all domains.
   - Target 10–20 % sequence-length reduction relative to regex baseline and Byte-BPE.
   - Confirm Byte-BPE invertibility but with higher token counts; WordPiece/BPE should show lower fidelity on structured corpora.

4. **Run ablations** by editing `experiments/run_experiments.py`:
   - Remove or customize regex priors per domain to quantify boundary violations.
   - Adjust `CandidateExtractorConfig.max_vocab` or `.weights` to replicate vocabulary-size and scoring-weight sweeps.
   - Toggle `--no-baselines` for ablation-only runs focused on DST variants.

5. **Document findings** back into the paper by updating the benchmark table (`design/paper.tex`, Table~\ref{tab:main}) and the ablation table (`Table~\ref{tab:ablation}`) with fresh metrics.

## Suggested workflow

- Use `uv run python -m experiments.run_experiments ... --output-dir artifacts/<domain>` to materialize artifacts for downstream models.
- Load `artifacts/<domain>/tokenizer.json` into Hugging Face `PreTrainedTokenizerFast.from_pretrained(artifacts/<domain>)` when evaluating LLMs.
- Add new domain recipes by extending `DOMAIN_RECIPES` in `experiments/run_experiments.py` with custom normalizers and regex hints.
- When iterating on tokenizer scoring weights, regenerate JSON summaries so the deltas can be plotted or pasted into the manuscript.

## References

- Paper draft: `design/paper.tex`
- DST core implementation: `src/dst/pipeline.py`, `src/dst/tokenizer.py`
- Experiment harness usage: `python -m experiments.run_experiments --help`
