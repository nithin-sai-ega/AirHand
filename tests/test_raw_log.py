"""Tests for verbatim packet logging: the body must always round-trip byte-for-byte."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from airhand.ingest.raw_log import (
    BODY_ENCODING_BASE64,
    BODY_ENCODING_UTF8,
    RawPacketLog,
    decode_body,
    encode_body,
    envelope_from_json_line,
    envelope_to_json_line,
    read_raw_log,
)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b'{"payload": [{"name": "accelerometer", "values": {"x": 0.1}}]}',
        "temperature °C, été".encode(),
        b"line1\r\nline2\n",
    ],
)
def test_utf8_body_is_stored_as_text_and_round_trips(body: bytes) -> None:
    encoding, text = encode_body(body)
    assert encoding == BODY_ENCODING_UTF8
    assert decode_body(encoding, text) == body


@pytest.mark.parametrize(
    "body",
    [
        bytes(range(256)),  # every byte value, invalid as UTF-8
        b"\xff\xfe\x00\x01",
        b"\x1f\x8b\x08\x00",  # gzip magic number
    ],
)
def test_binary_body_falls_back_to_base64_and_round_trips(body: bytes) -> None:
    encoding, text = encode_body(body)
    assert encoding == BODY_ENCODING_BASE64
    assert decode_body(encoding, text) == body


def test_unknown_body_encoding_is_rejected() -> None:
    with pytest.raises(ValueError):
        decode_body("rot13", "abc")


def test_envelope_json_line_round_trip(tmp_path: Path) -> None:
    with RawPacketLog(tmp_path / "log.jsonl") as log:
        written = log.append(
            recv_wall_ns=1_700_000_000_000_000_000,
            recv_mono_ns=123_456_789,
            transport="http",
            peer="192.168.1.20:51234",
            body=b'{"a": 1}',
            method="POST",
            path="/data",
            headers={"Content-Type": "application/json"},
        )
    line = envelope_to_json_line(written)
    assert "\n" not in line
    assert envelope_from_json_line(line) == written


def test_log_file_preserves_order_and_bodies(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    bodies = [b"first", b"\x00\xffbinary", b"", b"last"]
    with RawPacketLog(path) as log:
        for i, body in enumerate(bodies):
            log.append(
                recv_wall_ns=i, recv_mono_ns=i, transport="udp", peer="10.0.0.2:5000", body=body
            )
        assert log.count == len(bodies)

    read_back = list(read_raw_log(path))
    assert [e.body for e in read_back] == bodies
    assert [e.seq for e in read_back] == list(range(len(bodies)))
    assert all(e.method is None and e.headers == {} for e in read_back)


def test_concurrent_appends_get_unique_sequence_numbers(tmp_path: Path) -> None:
    """HTTP and UDP receivers write from different threads; no line may be lost or mixed."""
    path = tmp_path / "log.jsonl"
    n_threads, per_thread = 8, 200

    with RawPacketLog(path) as log:

        def worker(thread_id: int) -> None:
            for i in range(per_thread):
                log.append(
                    recv_wall_ns=i,
                    recv_mono_ns=i,
                    transport="udp",
                    peer=f"10.0.0.{thread_id}:1",
                    body=f"{thread_id}-{i}".encode(),
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    envelopes = list(read_raw_log(path))
    assert len(envelopes) == n_threads * per_thread
    assert sorted(e.seq for e in envelopes) == list(range(n_threads * per_thread))
    expected_bodies = {f"{t}-{i}".encode() for t in range(n_threads) for i in range(per_thread)}
    assert {e.body for e in envelopes} == expected_bodies
