"""HTTP and UDP receivers for the Phase 0 feasibility probe.

These receivers deliberately do not interpret payloads. They timestamp each packet as soon
as its body has been fully read, write it verbatim to a RawPacketLog, and optionally call
a callback (used by the CLI to print a preview).

Standard-library servers are used on purpose: the probe should be small enough to trust at
a glance. The production ingest path (Phase 1) is a separate, async implementation.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from airhand.ingest.raw_log import PacketEnvelope, RawPacketLog

PacketCallback = Callable[[PacketEnvelope], None]

# How often the UDP thread wakes up to check whether it should stop.
_UDP_POLL_TIMEOUT_S = 0.2


def _format_peer(address: tuple[str, int]) -> str:
    host, port = address[0], address[1]
    return f"{host}:{port}"


class _ProbeHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries the log and callback for its handlers."""

    # Handler threads should not keep the process alive after Ctrl+C.
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        packet_log: RawPacketLog,
        on_packet: PacketCallback | None,
    ) -> None:
        super().__init__(address, _ProbeHTTPHandler)
        self.packet_log = packet_log
        self.on_packet = on_packet


class _ProbeHTTPHandler(BaseHTTPRequestHandler):
    """Accept any method on any path, log the body verbatim, answer 200 OK.

    We do not know in advance which method or path the phone app uses, so all are accepted
    and recorded. HTTP/1.1 is used so the app can keep its connection open between packets
    (closer to how it will behave in the real pipeline).
    """

    server: _ProbeHTTPServer
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (name required by BaseHTTPRequestHandler)
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def _read_body(self) -> bytes:
        """Read the request body using either chunked encoding or Content-Length."""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            return self._read_chunked_body()
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def _read_chunked_body(self) -> bytes:
        """Chunked format: '<hex size>\\r\\n<data>\\r\\n' repeated, ending with size 0."""
        parts: list[bytes] = []
        while True:
            size_line = self.rfile.readline()
            size = int(size_line.split(b";")[0].strip(), 16)
            if size == 0:
                # Skip optional trailer headers up to the final empty line.
                while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            parts.append(self.rfile.read(size))
            self.rfile.readline()  # CRLF after each chunk

    def _handle(self) -> None:
        body = self._read_body()
        # Timestamps are taken after the whole body has arrived: that is the earliest moment
        # the data could be used downstream.
        recv_wall_ns = time.time_ns()
        recv_mono_ns = time.perf_counter_ns()

        envelope = self.server.packet_log.append(
            recv_wall_ns=recv_wall_ns,
            recv_mono_ns=recv_mono_ns,
            transport="http",
            peer=_format_peer(self.client_address),
            body=body,
            method=self.command,
            path=self.path,
            headers=dict(self.headers.items()),
        )

        response = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

        if self.server.on_packet is not None:
            self.server.on_packet(envelope)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request stderr logging; the CLI prints its own preview."""


class ProbeServers:
    """Run an HTTP receiver and/or a UDP receiver on background threads.

    Pass port 0 to let the OS choose a free port (used by tests); the chosen ports are then
    available from http_address / udp_address after start().
    """

    def __init__(
        self,
        packet_log: RawPacketLog,
        *,
        bind_host: str,
        http_port: int | None,
        udp_port: int | None,
        udp_max_datagram_bytes: int,
        on_packet: PacketCallback | None = None,
    ) -> None:
        self._packet_log = packet_log
        self._bind_host = bind_host
        self._http_port = http_port
        self._udp_port = udp_port
        self._udp_max_datagram_bytes = udp_max_datagram_bytes
        self._on_packet = on_packet

        self._http_server: _ProbeHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._udp_socket: socket.socket | None = None
        self._udp_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def http_address(self) -> tuple[str, int] | None:
        if self._http_server is None:
            return None
        host, port = self._http_server.server_address[:2]
        return str(host), int(port)

    @property
    def udp_address(self) -> tuple[str, int] | None:
        if self._udp_socket is None:
            return None
        host, port = self._udp_socket.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._http_port is not None:
            self._http_server = _ProbeHTTPServer(
                (self._bind_host, self._http_port), self._packet_log, self._on_packet
            )
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever, name="probe-http", daemon=True
            )
            self._http_thread.start()

        if self._udp_port is not None:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.bind((self._bind_host, self._udp_port))
            self._udp_socket.settimeout(_UDP_POLL_TIMEOUT_S)
            self._udp_thread = threading.Thread(
                target=self._udp_loop, name="probe-udp", daemon=True
            )
            self._udp_thread.start()

    def _udp_loop(self) -> None:
        assert self._udp_socket is not None
        while not self._stop_event.is_set():
            try:
                body, address = self._udp_socket.recvfrom(self._udp_max_datagram_bytes)
            except TimeoutError:
                continue  # no data yet; loop to re-check the stop flag
            except ConnectionResetError:
                # Windows can report an ICMP "port unreachable" from an earlier exchange as
                # an error on the next recvfrom. It says nothing about incoming data.
                continue
            except OSError:
                if self._stop_event.is_set():
                    return  # socket closed by stop()
                raise
            recv_wall_ns = time.time_ns()
            recv_mono_ns = time.perf_counter_ns()
            envelope = self._packet_log.append(
                recv_wall_ns=recv_wall_ns,
                recv_mono_ns=recv_mono_ns,
                transport="udp",
                peer=_format_peer(address),
                body=body,
            )
            if self._on_packet is not None:
                self._on_packet(envelope)

    def stop(self) -> None:
        self._stop_event.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._http_thread is not None:
            self._http_thread.join()
        if self._udp_thread is not None:
            self._udp_thread.join()
        if self._udp_socket is not None:
            self._udp_socket.close()
