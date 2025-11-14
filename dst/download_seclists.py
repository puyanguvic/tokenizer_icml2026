#!/usr/bin/env python3
"""
dst/download_seclists.py

Clone SecLists (shallow) and copy all fuzzing .txt payload files to an output directory.

This script:
- Uses `git clone --depth 1` into a temporary directory (or reuses a local cache if provided).
- Copies all files under the `Fuzzing` tree that match `*.txt` to the output directory.
- Prints progress and a final summary.

Usage:
  python dst/download_seclists.py --out-dir datasets/seclists_raw
  python dst/download_seclists.py --out-dir /data/seclists --no-clean
  python dst/download_seclists.py --out-dir datasets/seclists_raw --pattern "*XSS*.txt"

Notes:
- Requires `git` available in PATH.
- The script is intentionally minimal: it does NOT perform any payload parsing or JSONL conversion.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO_URL = "https://github.com/danielmiessler/SecLists.git"
DEFAULT_PATTERN = "*.txt"


def clone_repo(shallow_dir: Path, repo_url: str = REPO_URL, depth: int = 1, verbose: bool = False) -> Path:
    """
    Clone the repo shallowly into shallow_dir. Returns the path to the cloned repo.
    If shallow_dir already exists and looks like a clone, reuse it.
    """
    shallow_dir = shallow_dir.resolve()
    if shallow_dir.exists() and any(shallow_dir.iterdir()):
        if verbose:
            print(f"[info] Using existing directory {shallow_dir} (will attempt to reuse).")
        return shallow_dir

    shallow_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), repo_url, str(shallow_dir)]
    try:
        print(f"[git] Cloning {repo_url} -> {shallow_dir} ... (this may take a minute)")
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError("`git` not found on PATH. Please install git or run on a machine with git.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed: {e}")
    return shallow_dir


def gather_fuzz_files(clone_dir: Path, pattern: str = DEFAULT_PATTERN) -> List[Path]:
    """
    Find all files under clone_dir/Fuzzing that match the glob pattern (default *.txt).
    Returns a sorted list of Paths.
    """
    fuzz_root = clone_dir / "Fuzzing"
    if not fuzz_root.exists():
        # some repo layouts could vary: try case-insensitive fallback
        raise RuntimeError(f"Fuzzing directory not found in cloned repo at {fuzz_root}")
    matched = sorted(fuzz_root.rglob(pattern))
    # Filter to files only
    files = [p for p in matched if p.is_file()]
    return files


def copy_files_to_out(files: List[Path], out_dir: Path, overwrite: bool = False) -> List[Path]:
    """
    Copy given files to out_dir (flattened: keep only filename).
    Returns list of destination paths actually written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for src in files:
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            # skip existing file
            print(f"[skip] {dst.name} (exists)")
            continue
        shutil.copy2(src, dst)
        written.append(dst)
        print(f"[copied] {src.relative_to(src.parents[2]) if len(src.parents) >= 3 else src.name} -> {dst}")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download SecLists Fuzzing payloads (clone + copy).")
    parser.add_argument("--out-dir", required=True, help="Output directory to store *.txt files (will be created).")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache directory to store cloned repo (reuses if exists).")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern to match fuzz files (default: '*.txt').")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove temporary clone dir (useful for debugging).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out-dir.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    temp_dir = None
    try:
        # choose clone target: either user-provided cache or a temp dir
        if cache_dir:
            clone_target = cache_dir
            if not clone_target.exists() or not any(clone_target.iterdir()):
                # perform clone into cache_dir
                clone_repo(clone_target, verbose=args.verbose)
            else:
                if args.verbose:
                    print(f"[info] Reusing existing cache at {clone_target}")
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="seclists_clone_"))
            clone_target = clone_repo(temp_dir, verbose=args.verbose)

        # gather fuzz files
        try:
            files = gather_fuzz_files(clone_target, pattern=args.pattern)
        except RuntimeError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

        if not files:
            print("[warning] No files matched the pattern under Fuzzing/; try a different --pattern or inspect the repo layout.")
            return 3

        print(f"[info] Found {len(files)} files matching pattern '{args.pattern}' under {clone_target / 'Fuzzing'}")
        written = copy_files_to_out(files, out_dir, overwrite=args.overwrite)
        print(f"\n✅ Completed: copied {len(written)} files to {out_dir}")
        if args.verbose:
            print("Files copied:")
            for p in written:
                print("  -", p)
    finally:
        if temp_dir and not args.no_clean:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
