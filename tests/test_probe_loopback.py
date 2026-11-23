"""End-to-end loopback tests: send real HTTP/UDP traffic to the probe on 127.0.0.1."""

from __future__ import annotations

import http.client
import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from airhand.ingest.probe_server import ProbeServers
from airhand.ingest.raw_log import PacketEnvelope, RawPacketLog, read_raw_log

WAIT_TIMEOUT_S = 5.0


@pytest.fixture
def probe(tmp_path: Path) -> Iterator[tuple[ProbeServers, Path]]:
    """Probe on OS-chosen free ports (port 0) so tests never clash with a running probe."""
    log_path = tmp_path / "packets.jsonl"
    with RawPacketLog(log_path) as log:
        servers = ProbeServers(
            log,
            bind_host="127.0.0.1",
            http_port=0,
            udp_port=0,
            udp_max_datagram_bytes=65535,
        )
        servers.start()
        try:
            yield servers, log_path
        finally:
            servers.stop()


def wait_for_packets(log_path: Path, n: int) -> list[PacketEnvelope]:
    """Poll the log file until n packets are present (UDP has no acknowledgement)."""
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        envelopes = list(read_raw_log(log_path))
        if len(envelopes) >= n:
            return envelopes
        time.sleep(0.01)
    raise AssertionError(f"Expected {n} packets, got {len(list(read_raw_log(log_path)))}")


def http_request(
    address: tuple[str, int], method: str, path: str, body: bytes, *, chunked: bool = False
) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(address[0], address[1], timeout=WAIT_TIMEOUT_S)
    try:
        if chunked:
            # Send the body in three pieces using Transfer-Encoding: chunked.
            pieces = [body[:5], body[5:12], body[12:]]
            conn.request(method, path, body=iter(pieces), encode_chunked=True)
        else:
            conn.request(method, path, body=body)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_http_post_is_logged_verbatim(probe: tuple[ProbeServers, Path]) -> None:
    servers, log_path = probe
    assert servers.http_address is not None
    body = b'{"messageId": 7, "payload": [{"name": "gyroscope", "time": 1}]}'

    status, reply = http_request(servers.http_address, "POST", "/data", body)

    assert status == 200 and reply == b"ok"
    [envelope] = wait_for_packets(log_path, 1)
    assert envelope.transport == "http"
    assert envelope.method == "POST"
    assert envelope.path == "/data"
    assert envelope.body == body
    assert envelope.peer.startswith("127.0.0.1:")
    assert envelope.headers["Content-Length"] == str(len(body))


def test_http_chunked_body_is_reassembled(probe: tuple[ProbeServers, Path]) -> None:
    servers, log_path = probe
    assert servers.http_address is not None
    body = b"0123456789abcdefghijklmnopqrstuvwxyz"

    status, _ = http_request(servers.http_address, "PUT", "/any/path", body, chunked=True)

    assert status == 200
    [envelope] = wait_for_packets(log_path, 1)
    assert envelope.method == "PUT"
    assert envelope.body == body


def test_udp_datagrams_are_logged_verbatim_in_order(probe: tuple[ProbeServers, Path]) -> None:
    servers, log_path = probe
    assert servers.udp_address is not None
    datagrams = [b"1700000000.01,0.1,0.2,9.8", b"\x00\x01\x02\xff binary", b"third"]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for d in datagrams:
            sender.sendto(d, servers.udp_address)
            # Loopback UDP preserves order in practice; the short pause keeps this test
            # independent of socket buffer behaviour.
            time.sleep(0.01)

    envelopes = wait_for_packets(log_path, len(datagrams))
    assert [e.body for e in envelopes] == datagrams
    assert all(e.transport == "udp" and e.method is None for e in envelopes)
    mono = [e.recv_mono_ns for e in envelopes]
    assert mono == sorted(mono), "monotonic receive timestamps must not go backwards"


def test_http_and_udp_together(probe: tuple[ProbeServers, Path]) -> None:
    servers, log_path = probe
    assert servers.http_address is not None and servers.udp_address is not None

    http_request(servers.http_address, "POST", "/data", b"via-http")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(b"via-udp", servers.udp_address)

    envelopes = wait_for_packets(log_path, 2)
    assert {(e.transport, e.body) for e in envelopes} == {
        ("http", b"via-http"),
        ("udp", b"via-udp"),
    }
    assert sorted(e.seq for e in envelopes) == [0, 1]
