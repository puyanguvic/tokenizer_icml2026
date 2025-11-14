#!/usr/bin/env python3
"""
Utilities for building HTTP corpora suitable for DST tokenizer training from
Common Crawl WARC archives and CIC-IDS PCAP captures.

The resulting JSONL records align with the structure used in
`datasets/http_corpus/seclists_http.jsonl`:
    {
        "source": "commoncrawl" | "cicids",
        "method": "...",
        "url": "...",
        "protocol": "HTTP/1.1",
        "headers": {"Header": "Value", ...},
        "body": "...",
        "... dataset-specific metadata ..."
    }
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import dpkt
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


DEFAULT_HTTP_PORTS = (80, 8080, 8000, 8181)


def ensure_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return str(value)


def header_case(name: str) -> str:
    if not name:
        return name
    return "-".join(part.capitalize() for part in name.split("-"))


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz" or path.suffixes[-1:] == [".gz"]:
        return gzip.open(path, "rb")
    return open(path, "rb")


def request_to_record(
    request: dpkt.http.Request,
    *,
    source: str,
    target_uri: Optional[str] = None,
    scheme: str = "http",
    client_ip: Optional[str] = None,
    server_ip: Optional[str] = None,
    client_port: Optional[int] = None,
    server_port: Optional[int] = None,
    max_body_bytes: Optional[int] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    uri_text = ensure_text(request.uri) or "/"
    method = ensure_text(request.method).upper() or "GET"
    version = ensure_text(request.version) or "1.1"

    headers = {}
    host_header = ""
    for key, value in request.headers.items():
        key_text = header_case(ensure_text(key))
        val_text = ensure_text(value)
        headers[key_text] = val_text
        if key_text.lower() == "host":
            host_header = val_text

    # Determine absolute URL.
    if uri_text.startswith(("http://", "https://")):
        url = uri_text
    else:
        scheme_candidate = scheme
        netloc = host_header
        if not netloc and target_uri:
            tgt = urlsplit(target_uri)
            netloc = tgt.netloc
            scheme_candidate = tgt.scheme or scheme_candidate
            if uri_text == "/" and tgt.path:
                uri_text = tgt.path + (("?" + tgt.query) if tgt.query else "")
        if not netloc and server_ip:
            port_suffix = f":{server_port}" if server_port else ""
            netloc = f"{server_ip}{port_suffix}"

        parsed_uri = urlsplit(uri_text)
        path = parsed_uri.path or "/"
        query = parsed_uri.query
        fragment = parsed_uri.fragment
        if netloc:
            url = urlunsplit((scheme_candidate or "http", netloc, path, query, fragment))
        else:
            # Fall back to absolute-path form when Host header is missing.
            url = urlunsplit(("", "", path, query, fragment))

    body = request.body or b""
    truncated = False
    if max_body_bytes is not None and len(body) > max_body_bytes:
        body = body[:max_body_bytes]
        truncated = True
    body_text = ensure_text(body)

    http_lines = [f"{method} {uri_text} HTTP/{version}"]
    for header_name, header_value in headers.items():
        http_lines.append(f"{header_name}: {header_value}")
    if body_text:
        http_lines.append("")
        http_lines.append(body_text)
    http_message = "\r\n".join(http_lines)

    record: Dict[str, object] = {
        "source": source,
        "method": method,
        "uri": uri_text,
        "url": url,
        "protocol": f"HTTP/{version}",
        "headers": headers,
        "body": body_text,
        "text": http_message,
    }

    if truncated:
        record["body_truncated"] = True
    if client_ip:
        record["client_ip"] = client_ip
    if server_ip:
        record["server_ip"] = server_ip
    if client_port is not None:
        record["client_port"] = client_port
    if server_port is not None:
        record["server_port"] = server_port
    if extra:
        record.update(extra)
    return record


def iter_commoncrawl_requests(
    warc_paths: Sequence[Path],
    *,
    progress: bool = False,
    max_body_bytes: Optional[int] = 4096,
) -> Iterator[Dict[str, object]]:
    for warc_path in warc_paths:
        with open_maybe_gzip(warc_path) as fh:
            iterator = ArchiveIterator(fh)
            if progress:
                iterator = tqdm(iterator, desc=f"CommonCrawl {warc_path.name}", unit="record")
            for record in iterator:
                if record.rec_type != "request":
                    continue
                try:
                    raw = record.content_stream().read()
                    if not raw:
                        continue
                    http_request = dpkt.http.Request(raw)
                except (dpkt.NeedData, dpkt.UnpackError):
                    continue

                target_uri = record.rec_headers.get_header("WARC-Target-URI")
                extra_meta = {
                    "warc_file": str(warc_path),
                    "warc_record_id": record.rec_headers.get_header("WARC-Record-ID"),
                    "target_uri": target_uri,
                }
                yield request_to_record(
                    http_request,
                    source="commoncrawl",
                    target_uri=target_uri,
                    scheme=urlsplit(target_uri).scheme if target_uri else "http",
                    max_body_bytes=max_body_bytes,
                    extra=extra_meta,
                )


def _consume_http_messages(buffer: bytearray) -> Tuple[List[bytes], bytearray]:
    """
    Split accumulated TCP payload into complete HTTP requests.

    Returns:
        messages: list of raw HTTP request bytes (headers + optional body)
        leftover: bytes that do not yet form a complete message
    """
    data = bytes(buffer)
    messages: List[bytes] = []
    pos = 0
    length = len(data)
    while True:
        header_end = data.find(b"\r\n\r\n", pos)
        if header_end == -1:
            break
        line_end = data.find(b"\r\n", pos)
        if line_end == -1 or line_end > header_end:
            break
        start_line = data[pos:line_end]
        if start_line.startswith(b"HTTP/"):
            pos = header_end + 4
            continue
        if b" " not in start_line:
            pos = header_end + 4
            continue

        headers_block = data[line_end + 2 : header_end]
        content_length = 0
        chunked = False
        for header_line in headers_block.split(b"\r\n"):
            if b":" not in header_line:
                continue
            name, value = header_line.split(b":", 1)
            name_l = name.strip().lower()
            value_bytes = value.strip()
            if name_l == b"content-length":
                try:
                    content_length = int(value_bytes.decode("ascii", errors="ignore").strip() or "0")
                except ValueError:
                    content_length = 0
            elif name_l == b"transfer-encoding" and b"chunked" in value_bytes.lower():
                chunked = True

        if chunked:
            # Chunked requests are rare; skip to avoid partial decodes.
            pos = header_end + 4
            continue

        message_end = header_end + 4 + content_length
        if length < message_end:
            break

        messages.append(data[pos:message_end])
        pos = message_end

    leftover = bytearray(data[pos:])
    return messages, leftover


def iter_cicids_requests(
    pcap_paths: Sequence[Path],
    *,
    http_ports: Sequence[int] = DEFAULT_HTTP_PORTS,
    progress: bool = False,
    max_body_bytes: Optional[int] = 4096,
) -> Iterator[Dict[str, object]]:
    ports_set = set(http_ports)
    for pcap_path in pcap_paths:
        streams: Dict[Tuple[bytes, bytes, int, int], bytearray] = defaultdict(bytearray)
        fh = open_maybe_gzip(pcap_path)
        try:
            try:
                reader = dpkt.pcap.Reader(fh)
            except (ValueError, OSError):
                fh.close()
                fh = open(pcap_path, "rb")
                reader = dpkt.pcap.Reader(fh)

            iterator = reader
            if progress:
                iterator = tqdm(iterator, desc=f"CIC-IDS {pcap_path.name}", unit="packet")

            for timestamp, buf in iterator:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except (dpkt.NeedData, ValueError):
                    continue
                ip = eth.data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue
                tcp = ip.data
                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue
                if not tcp.data:
                    continue

                server_port = tcp.dport
                if server_port not in ports_set:
                    continue

                key = (ip.src, ip.dst, tcp.sport, tcp.dport)
                buffer = streams[key]
                buffer.extend(tcp.data)
                messages, leftover = _consume_http_messages(buffer)
                streams[key] = leftover

                if len(streams[key]) > 512_000:
                    streams[key] = bytearray()

                for message in messages:
                    try:
                        http_request = dpkt.http.Request(message)
                    except (dpkt.NeedData, dpkt.UnpackError):
                        continue

                    if isinstance(ip, dpkt.ip6.IP6):
                        client_ip = socket.inet_ntop(socket.AF_INET6, key[0])
                        server_ip = socket.inet_ntop(socket.AF_INET6, key[1])
                    else:
                        client_ip = socket.inet_ntoa(key[0])
                        server_ip = socket.inet_ntoa(key[1])

                    yield request_to_record(
                        http_request,
                        source="cicids",
                        client_ip=client_ip,
                        server_ip=server_ip,
                        client_port=key[2],
                        server_port=key[3],
                        max_body_bytes=max_body_bytes,
                        extra={
                            "pcap_file": str(pcap_path),
                            "timestamp": timestamp,
                        },
                    )
        finally:
            fh.close()


def write_records(
    records: Iterable[Dict[str, object]],
    out_path: Path,
    *,
    limit: Optional[int] = None,
    append: bool = False,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with out_path.open(mode, encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HTTP corpora for DST tokenizer training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cc = subparsers.add_parser("commoncrawl", help="Extract HTTP request records from Common Crawl WARC files.")
    cc.add_argument("warc", type=Path, nargs="+", help="Paths to WARC or WARC.GZ files.")
    cc.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    cc.add_argument("--limit", type=int, default=0, help="Limit the number of records written.")
    cc.add_argument("--append", action="store_true", help="Append to the output file instead of overwriting.")
    cc.add_argument("--max-body-bytes", type=int, default=4096, help="Truncate request bodies beyond this many bytes.")
    cc.add_argument("--progress", action="store_true", help="Show tqdm progress while reading.")

    cic = subparsers.add_parser("cicids", help="Extract HTTP request records from CIC-IDS PCAP dumps.")
    cic.add_argument("pcap", type=Path, nargs="+", help="Paths to PCAP or PCAP.GZ files.")
    cic.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    cic.add_argument("--limit", type=int, default=0, help="Limit the number of records written.")
    cic.add_argument("--append", action="store_true", help="Append to the output file instead of overwriting.")
    cic.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=list(DEFAULT_HTTP_PORTS),
        help="TCP destination ports treated as HTTP (default: %(default)s).",
    )
    cic.add_argument("--max-body-bytes", type=int, default=4096, help="Truncate request bodies beyond this many bytes.")
    cic.add_argument("--progress", action="store_true", help="Show tqdm progress while reading packets.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "commoncrawl":
        records_iter = iter_commoncrawl_requests(
            args.warc,
            progress=args.progress,
            max_body_bytes=args.max_body_bytes,
        )
    elif args.command == "cicids":
        records_iter = iter_cicids_requests(
            args.pcap,
            http_ports=args.ports,
            progress=args.progress,
            max_body_bytes=args.max_body_bytes,
        )
    else:
        parser.error("Unsupported command")
        return 1

    limit = args.limit if args.limit and args.limit > 0 else None
    written = write_records(records_iter, args.out, limit=limit, append=args.append)
    print(f"Wrote {written} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
