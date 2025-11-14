#!/usr/bin/env python3
"""
dst/download_warc.py

Download a subset of Common Crawl WARC files given a .paths.gz file
(e.g. warc.paths.gz, wat.paths.gz, wet.paths.gz) and save locally.

Dependencies:
    pip install requests tqdm
"""

from __future__ import annotations
import gzip
import random
from pathlib import Path
from typing import Iterable, Optional
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def download_warc_files(
    paths_gz_url: str,
    out_dir: str | Path,
    num_files: int = 5,
    sample: bool = True,
    workers: int = 4,
    base_url: str = "https://data.commoncrawl.org/",
    timeout: int = 60,
) -> list[Path]:
    """
    Download a limited number of WARC/WET/WAT files from Common Crawl.

    Args:
        paths_gz_url:  full URL to *.paths.gz file, e.g.
                      https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-35/warc.paths.gz
        out_dir:       directory to save .warc.gz files
        num_files:     number of files to download (approx)
        sample:        if True, randomly sample from the list
        workers:       concurrent download threads
        base_url:      Common Crawl file prefix (default data.commoncrawl.org)
        timeout:       per-file timeout (seconds)

    Returns:
        List of downloaded file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare HTTP session with retry
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=(500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # fetch .paths.gz index
    print(f"Fetching index list from {paths_gz_url} ...")
    resp = session.get(paths_gz_url, timeout=timeout)
    resp.raise_for_status()

    lines = [l.strip() for l in gzip.decompress(resp.content).decode("utf-8").splitlines() if l.strip()]
    if sample and len(lines) > num_files:
        lines = random.sample(lines, num_files)
    else:
        lines = lines[:num_files]

    urls = [base_url.rstrip("/") + "/" + p for p in lines]
    print(f"Selected {len(urls)} files to download")

    downloaded: list[Path] = []

    def _download_one(url: str) -> Optional[Path]:
        fname = out_dir / Path(url).name
        if fname.exists() and fname.stat().st_size > 0:
            return fname
        try:
            r = session.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return fname
        except Exception as e:
            print(f"⚠️  failed {url}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, u): u for u in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            fpath = fut.result()
            if fpath:
                downloaded.append(fpath)

    print(f"✅ Downloaded {len(downloaded)} files to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    download_warc_files(
        paths_gz_url="https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-43/warc.paths.gz",
        out_dir="datasets/warcs",
        num_files=10,
        workers=127,
        sample=True,
    )
