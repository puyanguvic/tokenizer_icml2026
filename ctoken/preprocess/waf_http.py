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
    num_rows = batch.num_rows

    def const_str(value: str) -> pa.Array:
        return pa.array([value] * num_rows, type=pa.string())

    def col(name: str) -> pa.Array:
        return _fill_str(batch.column(batch.schema.get_field_index(name)))

    method = col("method")
    url = col("url")
    proto = col("protocol")
    headers = col("headers")
    body = col("body")
    label = col("label")

    # Element-wise concat in Arrow/C++ (fast). Keep structure to help tokenizer.
    text = pc.binary_join_element_wise(
        const_str("<METHOD> "),
        method,
        const_str("\n<URL> "),
        url,
        const_str("\n<PROT> "),
        proto,
        const_str("\n<HDR>\n"),
        headers,
        const_str("\n<BODY>\n"),
        body,
        const_str("\n"),
        "",
    )

    out = pa.RecordBatch.from_arrays(
        [pa.array(text, type=pa.string()), pa.array(label, type=pa.string())],
        names=["text", "label"],
    )
    return out
