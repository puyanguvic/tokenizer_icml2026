from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from .registry import PreprocessSpec, register


def _fill_str(arr: pa.Array) -> pa.Array:
    # Ensure string and fill null with empty string.
    arr2 = pc.cast(arr, pa.string()) if arr.type != pa.string() else arr
    return pc.fill_null(arr2, "")


@register(
    PreprocessSpec(
        name="waf_http_v2",
        required_columns=["method", "url", "protocol", "headers", "body", "label"],
        text_key="text",
        label_key="label",
    )
)
def preprocess_waf_http_v2(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Map WAF parquet rows to a single structured text field.

    Output columns: text, label
    """

    def col(name: str) -> pa.Array:
        return _fill_str(batch.column(batch.schema.get_field_index(name)))

    method = col("method")
    url = col("url")
    proto = col("protocol")
    headers = col("headers")
    body = col("body")
    label = col("label")

    # Element-wise concat in Arrow/C++ (fast). Keep structure to help tokenizer.
    parts = [
        pc.add(pc.add("<METHOD> ", method), "\n"),
        pc.add(pc.add("<URL> ", url), "\n"),
        pc.add(pc.add("<PROT> ", proto), "\n"),
        pc.add("<HDR>\n", headers),
        "\n",
        pc.add("<BODY>\n", body),
        "\n",
    ]

    text = pc.binary_join_element_wise(parts, "")

    out = pa.RecordBatch.from_arrays(
        [pa.array(text, type=pa.string()), pa.array(label, type=pa.string())],
        names=["text", "label"],
    )
    return out
