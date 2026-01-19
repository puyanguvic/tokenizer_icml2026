from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParquetSource:
    """A unified input source for all corpora: Parquet only."""

    path: str
    columns: Optional[List[str]] = None
    batch_size: int = 8192


def iter_record_batches(src: ParquetSource) -> Iterable[pa.RecordBatch]:
    """Stream record batches from a parquet file."""

    pf = pq.ParquetFile(src.path)
    for batch in pf.iter_batches(batch_size=src.batch_size, columns=src.columns):
        yield batch
