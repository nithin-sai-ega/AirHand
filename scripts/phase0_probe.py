"""Phase 0 feasibility probe: receive packets from the phone app and log them verbatim.

Usage (from the repository root, with the venv active):

    python scripts/phase0_probe.py --note "still on desk, 100 Hz"

Each run creates data/raw/probe/<YYYYmmdd-HHMMSS>/ containing:
    packets.jsonl  one JSON line per received packet (see airhand.ingest.raw_log)
    session.json   run metadata: note, config, start/end time, packet counts

Nothing is parsed here on purpose. The payload format is inspected by hand first, and the
parser is written afterwards from real packets.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from airhand.config import ProbeConfig
from airhand.ingest.probe_server import ProbeServers
from airhand.ingest.raw_log import BODY_ENCODING_UTF8, PacketEnvelope, RawPacketLog, encode_body

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "probe.yaml"


class TrafficCounter:
    """Counts packets and bytes per transport between console summaries (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packets: dict[str, int] = {}
        self._bytes: dict[str, int] = {}
        self.total_packets: dict[str, int] = {}

    def add(self, transport: str, n_bytes: int) -> None:
        with self._lock:
            self._packets[transport] = self._packets.get(transport, 0) + 1
            self._bytes[transport] = self._bytes.get(transport, 0) + n_bytes
            self.total_packets[transport] = self.total_packets.get(transport, 0) + 1

    def take_interval(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return (packets, bytes) since the previous call and reset the interval counts."""
        with self._lock:
            packets, n_bytes = self._packets, self._bytes
            self._packets, self._bytes = {}, {}
        return packets, n_bytes


def make_preview(envelope: PacketEnvelope, max_chars: int) -> str:
    """One console line describing a packet, with the start of its payload."""
    header = f"[{envelope.seq:6d} {envelope.transport:4s}"
    if envelope.method is not None:
        header += f" {envelope.method} {envelope.path}"
    header += f" {len(envelope.body)} B]"

    encoding, text = encode_body(envelope.body)
    if encoding == BODY_ENCODING_UTF8:
        snippet = text.replace("\r", "\\r").replace("\n", "\\n")
    else:
        snippet = "<binary> " + envelope.body[: max_chars // 2].hex(" ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return f"{header} {snippet}"


def lan_ipv4_addresses() -> list[str]:
    """IPv4 addresses of this machine (excluding loopback), to show the URL for the app."""
    try:
        _, _, addresses = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return []
    return sorted(a for a in addresses if not a.startswith("127."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--note", default="", help="What this recording is (posture, rate...)")
    parser.add_argument("--http-port", type=int, help="Override http_port from the config")
    parser.add_argument("--udp-port", type=int, help="Override udp_port from the config")
    parser.add_argument("--no-http", action="store_true", help="Do not start the HTTP receiver")
    parser.add_argument("--no-udp", action="store_true", help="Do not start the UDP receiver")
    parser.add_argument("--quiet", action="store_true", help="Only print rate summaries")
    return parser.parse_args()


def write_session_json(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = ProbeConfig.from_yaml(args.config)
    http_port = None if args.no_http else (args.http_port or config.http_port)
    udp_port = None if args.no_udp else (args.udp_port or config.udp_port)
    print_each_packet = config.print_each_packet and not args.quiet

    session_dir = REPO_ROOT / config.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    session_json = session_dir / "session.json"
    metadata: dict[str, object] = {
        "note": args.note,
        "started_at": datetime.now().astimezone().isoformat(),
        "config": asdict(config),
        "http_port": http_port,
        "udp_port": udp_port,
        "python": sys.version,
        "platform": platform.platform(),
        # Lets the analysis relate the two receive clocks recorded per packet.
        "clock_reference": {"wall_ns": time.time_ns(), "mono_ns": time.perf_counter_ns()},
    }
    # Written now as well as at the end, so metadata survives a crash.
    write_session_json(session_json, metadata)

    counter = TrafficCounter()

    def on_packet(envelope: PacketEnvelope) -> None:
        counter.add(envelope.transport, len(envelope.body))
        if print_each_packet:
            print(make_preview(envelope, config.preview_chars), flush=True)

    with RawPacketLog(session_dir / "packets.jsonl") as packet_log:
        servers = ProbeServers(
            packet_log,
            bind_host=config.bind_host,
            http_port=http_port,
            udp_port=udp_port,
            udp_max_datagram_bytes=config.udp_max_datagram_bytes,
            on_packet=on_packet,
        )
        servers.start()

        print(f"Logging to {session_dir}")
        ips = lan_ipv4_addresses() or ["<laptop-ip>"]
        if servers.http_address:
            for ip in ips:
                print(f"  HTTP: http://{ip}:{servers.http_address[1]}/data")
        if servers.udp_address:
            for ip in ips:
                print(f"  UDP : {ip}:{servers.udp_address[1]}")
        print("Press Ctrl+C to stop.\n", flush=True)

        try:
            while True:
                time.sleep(config.stats_interval_s)
                packets, n_bytes = counter.take_interval()
                parts = [
                    f"{t}: {packets[t] / config.stats_interval_s:.1f} pkt/s, "
                    f"{n_bytes[t] / config.stats_interval_s / 1024:.1f} KiB/s"
                    for t in sorted(packets)
                ]
                summary = " | ".join(parts) if parts else "no packets"
                print(f"-- last {config.stats_interval_s:g} s: {summary}", flush=True)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            servers.stop()

        metadata["ended_at"] = datetime.now().astimezone().isoformat()
        metadata["packet_counts"] = dict(counter.total_packets)
        metadata["packets_logged"] = packet_log.count
        write_session_json(session_json, metadata)

    print(f"Logged {metadata['packets_logged']} packets to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
