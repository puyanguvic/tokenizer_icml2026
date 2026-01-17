"""ctok-core: controlled tokenization runtime and induction."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ctok-core")
except PackageNotFoundError:  # pragma: no cover - runtime fallback
    __version__ = "0.0.0"

__all__ = ["__version__"]
