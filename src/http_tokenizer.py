from __future__ import annotations

import re
from typing import List


def http_clean_line(line: str) -> str:
    """
    Lightweight HTTP-specific normalization applied before tokenization.
    """
    line = line.strip().replace("\r", " ").replace("\n", " ")

    def _percent_decode(match: re.Match[str]) -> str:
        byte = bytes.fromhex(match.group(1))
        try:
            return byte.decode("utf-8")
        except UnicodeDecodeError:
            return byte.decode("latin-1")

    line = re.sub(r"%([0-9A-Fa-f]{2})", _percent_decode, line)
    return line


# Grammar priors used by the DST pipeline to guide candidate extraction.
HTTP_GRAMMAR_PATTERNS: List[str] = [
    r"https?://[^\s\"'<>]+",  # URLs
    r"[A-Za-z0-9\-\._]+=[^&\s]+",  # query parameters / key=value pairs
    r"[A-Za-z0-9\-\._]+:[^\s]+",  # header-style key: value
    r"(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+/[^\s]*",  # request line with path
    r"HTTP/\d\.\d",  # protocol versions
    r"\d{1,3}(?:\.\d{1,3}){3}",  # IPv4 addresses
    r"[A-Za-z0-9\-_]+\.(?:com|org|net|io|gov|edu|cn|de|uk)",  # hostnames
    r"[A-Fa-f0-9]{32,64}",  # hashes / identifiers
]

