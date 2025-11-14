#!/usr/bin/env python3
"""
dst/merge_seclists_corpus.py

Merge selected SecLists .txt files into one deduped payload file for tokenizer training.

Usage:
  python dst/merge_seclists_corpus.py \
    --src-dir datasets/seclists_raw \
    --out-file datasets/http_corpus/seclists_payloads.txt \
    --patterns SQL XSS LFI SSRF command XML JSON "special-chars" "User-Agents"

If --patterns omitted, uses a sensible default set for HTTP WAF.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Iterable, Set
import hashlib
import urllib.parse

DEFAULT_PATTERNS = [
    "SQL", "Databases", "XSS", "HTML5", "URI", "LFI", "command", "commix",
    "XXE", "XML", "JSON", "SSRF", "SSI", "template", "special-chars",
    "URI-hex", "doble-uri-hex", "file-extensions", "extension-test",
    "User-Agents", "http-request-methods", "naughty", "Unicode", "Metacharacters"
]

def iter_candidate_files(src_dir: Path, patterns: Iterable[str]) -> List[Path]:
    files = []
    for p in src_dir.rglob("*.txt"):
        name = p.name.lower()
        for pat in patterns:
            if pat.lower() in name:
                files.append(p)
                break
    return sorted(set(files))

def normalize_line(line: str) -> str:
    # strip, remove surrounding whitespace/newlines, collapse spaces
    s = line.strip()
    s = " ".join(s.split())
    return s

def should_keep(line: str, max_len: int = 2000) -> bool:
    if not line:
        return False
    if line.startswith("#") or line.startswith("//"):
        return False
    if len(line) > max_len:
        return False
    return True

def merge_files(files: List[Path], out_file: Path, limit_per_file: int = 0, add_url_encoded: bool = False) -> int:
    seen: Set[str] = set()
    written = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fout:
        for f in files:
            cnt = 0
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    line = normalize_line(raw)
                    if not should_keep(line):
                        continue
                    h = hashlib.sha1(line.encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    fout.write(line + "\n")
                    written += 1
                    cnt += 1
                    if add_url_encoded:
                        # also add url-encoded variant for param payload training
                        enc = urllib.parse.quote_plus(line)
                        eh = hashlib.sha1(enc.encode("utf-8")).hexdigest()
                        if eh not in seen:
                            seen.add(eh)
                            fout.write(enc + "\n")
                            written += 1
                    if limit_per_file and cnt >= limit_per_file:
                        break
    return written

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, help="Directory with SecLists .txt files (e.g. datasets/seclists_raw)")
    parser.add_argument("--out-file", required=True, help="Output merged payload file (one payload per line)")
    parser.add_argument("--patterns", nargs="*", default=None, help="Filename substrings to match (default sensible list)")
    parser.add_argument("--limit-per-file", type=int, default=0, help="Max lines per input file (0 = no limit)")
    parser.add_argument("--add-url-encoded", action="store_true", help="Also emit URL-encoded forms of payloads")
    args = parser.parse_args(argv)

    src = Path(args.src_dir)
    if not src.exists():
        print(f"Source dir not found: {src}")
        return 2

    patterns = args.patterns if args.patterns else DEFAULT_PATTERNS
    print("Using patterns:", patterns)
    files = iter_candidate_files(src, patterns)
    if not files:
        print("No files matched the patterns under", src)
        return 3
    print(f"Found {len(files)} candidate files. Example:\n  -", "\n  - ".join(p.name for p in files[:10]))

    out = Path(args.out_file)
    total = merge_files(files, out, limit_per_file=args.limit_per_file, add_url_encoded=args.add_url_encoded)
    print(f"✅ Wrote {total} unique payload lines to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
