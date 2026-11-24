"""Tests for the Sensor Logger message parser and probe-session loader.

The example messages use the exact structure and field names observed in real packets
captured in Phase 0; only the numbers are made up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from airhand.ingest.probe_session import load_probe_session
from airhand.ingest.raw_log import RawPacketLog
from airhand.ingest.sensor_logger import (
    WRIST_MOTION,
    WRIST_MOTION_FIELDS,
    WRIST_MOTION_UNITS,
    parse_message,
    to_chunk,
)

T0 = 1_795_463_355_248_168_700  # a real-looking epoch time in nanoseconds
PERIOD_NS = 10_000_000  # 100 Hz


def wrist_sample(time_ns: int, rotation_x: float = 0.0) -> dict[str, Any]:
    values = dict.fromkeys(WRIST_MOTION_FIELDS, 0.0)
    values["rotationRateX"] = rotation_x
    values["quaternionW"] = 1.0
    return {"name": WRIST_MOTION, "time": time_ns, "values": values}


def message_body(message_id: int, samples: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "messageId": message_id,
            "sessionId": "00000000-0000-4000-8000-00000000000a",
            "deviceId": "00000000-0000-4000-8000-00000000000b",
            "payload": samples,
        }
    ).encode()


def test_parse_message_reads_metadata_and_samples() -> None:
    body = message_body(
        17,
        [
            wrist_sample(T0, rotation_x=0.5),
            {"name": "gyroscope", "time": T0 + 1, "values": {"x": 0.1, "y": 0.2, "z": 0.3}},
        ],
    )
    message = parse_message(body)

    assert message.message_id == 17
    assert message.session_id == "00000000-0000-4000-8000-00000000000a"
    assert len(message.samples) == 2
    wrist, gyro = message.samples
    assert wrist.name == WRIST_MOTION
    assert wrist.time_ns == T0
    assert wrist.values["rotationRateX"] == 0.5
    assert set(wrist.values) == set(WRIST_MOTION_FIELDS)
    assert gyro.values == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_large_nanosecond_timestamps_are_exact() -> None:
    """Epoch nanoseconds exceed float precision; they must stay exact Python ints."""
    message = parse_message(message_body(1, [wrist_sample(T0 + 1)]))
    assert message.samples[0].time_ns == T0 + 1


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        (b"not json", "not valid JSON"),
        (b"[1, 2]", "JSON object"),
        (b'{"sessionId": "a", "deviceId": "b", "payload": []}', "messageId"),
        (b'{"messageId": 1, "sessionId": "a", "deviceId": "b"}', "payload"),
        (
            b'{"messageId": 1, "sessionId": "a", "deviceId": "b", "payload": [{"name": "x"}]}',
            "values",
        ),
    ],
)
def test_malformed_messages_raise_clear_errors(body: bytes, expected_error: str) -> None:
    with pytest.raises(ValueError, match=expected_error):
        parse_message(body)


def write_session(session_dir: Path, bodies: list[bytes]) -> None:
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({"note": "unit test"}), encoding="utf-8")
    with RawPacketLog(session_dir / "packets.jsonl") as log:
        # A browser connectivity check that the loader must ignore.
        log.append(
            recv_wall_ns=T0,
            recv_mono_ns=0,
            transport="http",
            peer="p",
            body=b"",
            method="GET",
            path="/test",
        )
        for i, body in enumerate(bodies):
            log.append(
                recv_wall_ns=T0 + (i + 1) * 1_000_000_000,
                recv_mono_ns=(i + 1) * 1_000_000_000,
                transport="http",
                peer="p",
                body=body,
                method="POST",
                path="/data",
            )


def test_load_probe_session_links_samples_to_packets(tmp_path: Path) -> None:
    session_dir = tmp_path / "20260101-000000"
    bodies = [
        message_body(10, [wrist_sample(T0 + k * PERIOD_NS) for k in range(0, 3)]),
        message_body(11, [wrist_sample(T0 + k * PERIOD_NS) for k in range(3, 5)]),
    ]
    write_session(session_dir, bodies)

    session = load_probe_session(session_dir)

    assert session.note == "unit test"
    assert list(session.packets["message_id"]) == [10, 11]
    assert list(session.packets["n_samples"]) == [3, 2]
    wrist = session.sensor(WRIST_MOTION)
    assert len(wrist) == 5
    assert list(wrist["time_ns"]) == [T0 + k * PERIOD_NS for k in range(5)]
    # Packet 0 was the GET; the two POSTs are packets 1 and 2.
    assert list(wrist["packet_seq"]) == [1, 1, 1, 2, 2]
    assert "rotationRateX" in wrist.columns


def test_to_chunk_selects_sensor_and_orders_channels() -> None:
    message = parse_message(
        message_body(
            3,
            [
                wrist_sample(T0, rotation_x=0.1),
                {"name": "gyroscope", "time": T0 + 5, "values": {"x": 9.0, "y": 9.0, "z": 9.0}},
                wrist_sample(T0 + PERIOD_NS, rotation_x=0.2),
            ],
        )
    )
    channels = ("quaternionW", "rotationRateX")
    chunk = to_chunk(
        message,
        sensor=WRIST_MOTION,
        channels=channels,
        packet_seq=4,
        recv_wall_ns=T0 + 999,
        recv_mono_ns=123,
    )

    assert chunk is not None
    assert chunk.packet_seq == 4 and chunk.recv_wall_ns == T0 + 999 and chunk.recv_mono_ns == 123
    assert chunk.channels == channels
    np.testing.assert_array_equal(chunk.time_ns, [T0, T0 + PERIOD_NS])
    np.testing.assert_array_equal(chunk.values, [[1.0, 0.1], [1.0, 0.2]])
    assert chunk.time_ns.dtype == np.int64


def test_to_chunk_returns_none_without_the_sensor() -> None:
    message = parse_message(
        message_body(1, [{"name": "gyroscope", "time": T0, "values": {"x": 0, "y": 0, "z": 0}}])
    )
    kwargs = {"packet_seq": 0, "recv_wall_ns": 0, "recv_mono_ns": 0}
    assert to_chunk(message, sensor=WRIST_MOTION, channels=("rotationRateX",), **kwargs) is None


def test_to_chunk_rejects_missing_channel() -> None:
    message = parse_message(message_body(1, [wrist_sample(T0)]))
    with pytest.raises(ValueError, match="missingField"):
        to_chunk(
            message,
            sensor=WRIST_MOTION,
            channels=("rotationRateX", "missingField"),
            packet_seq=0,
            recv_wall_ns=0,
            recv_mono_ns=0,
        )


def test_every_wrist_motion_field_has_a_unit() -> None:
    assert set(WRIST_MOTION_UNITS) == set(WRIST_MOTION_FIELDS)
