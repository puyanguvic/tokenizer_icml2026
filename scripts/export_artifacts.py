#!/usr/bin/env python3
"""Export tokenizer artifacts for release."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ctok artifacts.")
    parser.add_argument("--artifact", required=True, help="Artifact directory.")
    parser.add_argument("--output", required=True, help="Output archive path.")
    args = parser.parse_args()

    raise NotImplementedError("Implement artifact packaging.")


if __name__ == "__main__":
    raise SystemExit(main())
