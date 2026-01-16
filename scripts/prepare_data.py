#!/usr/bin/env python3
"""Dataset preparation entrypoint."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare datasets for ctok.")
    parser.add_argument("--input", required=True, help="Raw data input path.")
    parser.add_argument("--output", required=True, help="Processed data output path.")
    args = parser.parse_args()

    raise NotImplementedError("Implement dataset normalization and export.")


if __name__ == "__main__":
    raise SystemExit(main())
