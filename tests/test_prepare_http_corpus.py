import dpkt

from data_waf.prepare_http_corpus import (
    _consume_http_messages,
    request_to_record,
    write_records,
)


def test_request_to_record_builds_absolute_url_with_host():
    raw = b"GET /admin/login.php?next=%2F HTTP/1.1\r\nHost: example.com\r\nUser-Agent: test\r\n\r\n"
    request = dpkt.http.Request(raw)
    record = request_to_record(request, source="unit", target_uri=None)

    assert record["method"] == "GET"
    assert record["uri"] == "/admin/login.php?next=%2F"
    assert record["url"] == "http://example.com/admin/login.php?next=%2F"
    assert record["headers"]["Host"] == "example.com"
    assert record["protocol"] == "HTTP/1.1"
    assert record["text"].startswith("GET /admin/login.php?next=%2F HTTP/1.1")


def test_consume_http_messages_extracts_multiple_requests():
    payload = (
        b"GET /one HTTP/1.1\r\nHost: a\r\n\r\n"
        b"POST /two HTTP/1.1\r\nHost: b\r\nContent-Length: 3\r\n\r\nabc"
    )
    messages, leftover = _consume_http_messages(bytearray(payload))
    assert len(messages) == 2
    assert leftover == bytearray()

    first_request = dpkt.http.Request(messages[0])
    second_request = dpkt.http.Request(messages[1])
    assert first_request.method == "GET"
    assert second_request.method == "POST"
    assert second_request.body == b"abc"


def test_write_records_limits_output(tmp_path):
    records = [{"source": "unit", "method": "GET", "url": "http://x", "protocol": "HTTP/1.1", "headers": {}, "body": ""}]
    out_file = tmp_path / "out.jsonl"
    written = write_records(records * 5, out_file, limit=3)
    assert written == 3
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
