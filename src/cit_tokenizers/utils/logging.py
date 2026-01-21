from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def configure_logging(level: Optional[str] = None) -> None:
    """Configure stdlib logging once.

    Respects env var CIT_LOG_LEVEL if `level` is None.
    """
    lvl = (level or os.environ.get("CIT_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO), format=_DEFAULT_FORMAT)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
