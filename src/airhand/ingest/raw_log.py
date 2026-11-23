"""Verbatim packet logging.

The phone app's payload format is not under our control, so during the feasibility probe
nothing is parsed. Each received packet is wrapped in an "envelope" (receive timestamps,
transport details, raw body) and appended to a JSON Lines file, one packet per line.

The body is stored so that it round-trips to the exact original bytes:
- if it is valid UTF-8 it is stored as text (readable when you open the file),
- otherwise it is stored as base64.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

BODY_ENCODING_UTF8 = "utf-8"
BODY_ENCODING_BASE64 = "base64"


@dataclass(frozen=True)
class PacketEnvelope:
    """One received packet plus the metadata needed to analyse timing later.

    Attributes:
        seq: Order in which the packet was logged, starting at 0.
        recv_wall_ns: Wall-clock receive time (time.time_ns). Comparable with phone/watch
            timestamps, but can jump if the OS adjusts its clock.
        recv_mono_ns: Monotonic receive time (time.perf_counter_ns). Use for intervals.
        transport: "http" or "udp".
        peer: Sender address as "ip:port".
        body: Raw payload bytes exactly as received.
        method: HTTP method (None for UDP).
        path: HTTP request path (None for UDP).
        headers: HTTP request headers (empty for UDP).
    """

    seq: int
    recv_wall_ns: int
    recv_mono_ns: int
    transport: str
    peer: str
    body: bytes
    method: str | None = None
    path: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def encode_body(body: bytes) -> tuple[str, str]:
    """Return (encoding, text) for storing raw bytes inside JSON.

    Strict UTF-8 decoding is lossless: if it succeeds, encoding the text again gives back
    the identical bytes. Anything that is not valid UTF-8 falls back to base64.
    """
    try:
        return BODY_ENCODING_UTF8, body.decode("utf-8")
    except UnicodeDecodeError:
        return BODY_ENCODING_BASE64, base64.b64encode(body).decode("ascii")


def decode_body(encoding: str, text: str) -> bytes:
    """Inverse of encode_body."""
    if encoding == BODY_ENCODING_UTF8:
        return text.encode("utf-8")
    if encoding == BODY_ENCODING_BASE64:
        return base64.b64decode(text)
    raise ValueError(f"Unknown body encoding: {encoding!r}")


def envelope_to_json_line(envelope: PacketEnvelope) -> str:
    """Serialise an envelope to one JSON line (no trailing newline)."""
    body_encoding, body_text = encode_body(envelope.body)
    record = {
        "seq": envelope.seq,
        "recv_wall_ns": envelope.recv_wall_ns,
        "recv_mono_ns": envelope.recv_mono_ns,
        "transport": envelope.transport,
        "peer": envelope.peer,
        "method": envelope.method,
        "path": envelope.path,
        "headers": envelope.headers,
        "body_len": len(envelope.body),
        "body_encoding": body_encoding,
        "body": body_text,
    }
    # ensure_ascii=False keeps non-ASCII text readable; the file itself is written as UTF-8.
    return json.dumps(record, ensure_ascii=False)


def envelope_from_json_line(line: str) -> PacketEnvelope:
    """Parse one JSON line back into an envelope."""
    record = json.loads(line)
    return PacketEnvelope(
        seq=record["seq"],
        recv_wall_ns=record["recv_wall_ns"],
        recv_mono_ns=record["recv_mono_ns"],
        transport=record["transport"],
        peer=record["peer"],
        body=decode_body(record["body_encoding"], record["body"]),
        method=record["method"],
        path=record["path"],
        headers=record["headers"],
    )


class RawPacketLog:
    """Append-only, thread-safe JSONL writer.

    The HTTP and UDP receivers run on different threads, so writes are serialised with a
    lock. Every line is flushed immediately: if the probe crashes or is killed, everything
    received up to that point is already on disk.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("a", encoding="utf-8", newline="\n")
        self._count = 0

    @property
    def count(self) -> int:
        """Number of packets written so far."""
        return self._count

    def append(
        self,
        *,
        recv_wall_ns: int,
        recv_mono_ns: int,
        transport: str,
        peer: str,
        body: bytes,
        method: str | None = None,
        path: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> PacketEnvelope:
        """Build an envelope, assign its sequence number and write it.

        The sequence number is assigned inside the lock so two threads can never get the
        same number. Returns the envelope that was written.
        """
        with self._lock:
            envelope = PacketEnvelope(
                seq=self._count,
                recv_wall_ns=recv_wall_ns,
                recv_mono_ns=recv_mono_ns,
                transport=transport,
                peer=peer,
                body=body,
                method=method,
                path=path,
                headers=dict(headers or {}),
            )
            self._file.write(envelope_to_json_line(envelope) + "\n")
            self._file.flush()
            self._count += 1
        return envelope

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> RawPacketLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_raw_log(path: Path) -> Iterator[PacketEnvelope]:
    """Yield envelopes from a JSONL log file in the order they were written."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield envelope_from_json_line(line)
