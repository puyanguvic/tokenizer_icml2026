#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


# =========================
# Config (edit these)
# =========================
REPO_ID = "puyang2025/waf_data_v2"
SPLIT_FILE = "train.parquet"          # 也可以换成 "heldout.parquet"
CACHE_DIR = "./_hf_cache_waf_v2"
WORK_DIR = "./_work_ctok_waf_v2"

BUILDER_PY = "ctok_core/build_ctok_unigram_artifact.py"

VOCAB_SIZE = 2048
MODEL_MAX_LEN = 512

OUT_ARTIFACT_DIR = "./ctok_waf_data_v2_unigram_2048"


def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def make_text(method, url, protocol, headers, body) -> str:
    # 结构化拼接：字段标签 + 换行，利于 tokenizer 学到边界
    return (
        f"<METHOD> {_safe_str(method)}\n"
        f"<URL> {_safe_str(url)}\n"
        f"<PROT> {_safe_str(protocol)}\n"
        f"<HDR>\n{_safe_str(headers)}\n"
        f"<BODY>\n{_safe_str(body)}\n"
    )


def export_text_parquet(src_parquet: str, dst_parquet: str, batch_rows: int = 8192) -> None:
    """
    Stream-read src parquet, build a new parquet with columns:
      - text (string)
      - label (string)
    """
    os.makedirs(Path(dst_parquet).parent, exist_ok=True)

    pf = pq.ParquetFile(src_parquet)

    writer = None
    try:
        for batch in pf.iter_batches(batch_size=batch_rows):
            # batch: RecordBatch
            cols = batch.schema.names
            # expected columns: method,url,protocol,headers,body,label
            # we build text + label
            method = batch.column(cols.index("method")).to_pylist()
            url = batch.column(cols.index("url")).to_pylist()
            protocol = batch.column(cols.index("protocol")).to_pylist()
            headers = batch.column(cols.index("headers")).to_pylist()
            body = batch.column(cols.index("body")).to_pylist()
            label = batch.column(cols.index("label")).to_pylist()

            texts = [
                make_text(m, u, p, h, b)
                for (m, u, p, h, b) in zip(method, url, protocol, headers, body)
            ]

            out_batch = pa.RecordBatch.from_arrays(
                [pa.array(texts, type=pa.string()), pa.array([_safe_str(y) for y in label], type=pa.string())],
                names=["text", "label"],
            )
            table = pa.Table.from_batches([out_batch])

            if writer is None:
                writer = pq.ParquetWriter(dst_parquet, table.schema, compression="zstd")
            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    # 1) Download parquet from HF Hub
    local_train = hf_hub_download(
        repo_id=REPO_ID,
        filename=SPLIT_FILE,
        repo_type="dataset",
        cache_dir=CACHE_DIR,
    )
    print(f"[OK] Downloaded: {local_train}")

    # 2) Build derived parquet with a single text column
    out_text_parquet = str(Path(WORK_DIR) / f"{Path(SPLIT_FILE).stem}_text.parquet")
    print(f"[RUN] Exporting text parquet -> {out_text_parquet}")
    export_text_parquet(local_train, out_text_parquet, batch_rows=8192)
    print(f"[OK] Wrote: {out_text_parquet}")

    # 3) Train tokenizer artifact using your builder
    builder = Path(BUILDER_PY)
    if not builder.exists():
        raise FileNotFoundError(
            f"Cannot find BUILDER_PY={BUILDER_PY}. Please set it to your unigram builder script path."
        )

    outdir = Path(OUT_ARTIFACT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(builder),
        "--format", "parquet",
        "--corpus", out_text_parquet,
        "--text_key", "text",
        "--outdir", str(outdir),
        "--vocab_size", str(VOCAB_SIZE),
        "--model_max_length", str(MODEL_MAX_LEN),
    ]
    print("[RUN] Training tokenizer artifact:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[OK] Artifact at: {outdir}")

    # 4) Verify standard HF loading (no trust_remote_code)
    print("[RUN] Verifying AutoTokenizer load ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(outdir))  # no trust_remote_code
    print("[OK] Loaded:", type(tok))
    print("[OK] backend_tokenizer:", type(tok.backend_tokenizer))

    print(tok.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
    print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("GET /a.php?id=12345&x=../etc/passwd HTTP/1.1"))

    sample = (
        "<METHOD> GET\n"
        "<URL> /blog/index.php/wp-json/oembed/HTTP%3A%2F%2Fwww.google.com%2F\n"
        "<PROT> HTTP/1.1\n"
        "<HDR>\nUser-Agent: Mozilla/5.0\r\nHost: test-site.com\r\n\n"
        "<BODY>\n\n"
    )
    print(tok.tokenize(sample)[:80])
    print("[DONE]")


if __name__ == "__main__":
    main()
