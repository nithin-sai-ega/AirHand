"""Parser for Sensor Logger HTTP push messages.

Written from real packets captured by the Phase 0 probe (Sensor Logger 1.69 on iOS,
Apple Watch companion app). Each HTTP POST body is one JSON object:

    {
      "messageId": 137,                  # increments by 1 per message; gaps mean lost messages
      "sessionId": "<uuid>",             # one per recording in the app
      "deviceId":  "<uuid>",
      "payload": [                       # many samples, possibly from several sensors, in order
        {"name": "wrist motion",
         "time": 1795463355248168700,    # nanoseconds since the Unix epoch
         "values": {"rotationRateX": -0.20, ..., "quaternionW": 0.71, ...}},
        ...
      ]
    }

Sensor names seen so far: "wrist motion" (the Apple Watch device-motion stream) and the
iPhone's own "accelerometer", "gyroscope", "orientation", "accelerometeruncalibrated",
"gyroscopeuncalibrated". Each sensor has its own set of value fields, so values are kept as
a name -> float mapping rather than fixed attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from airhand.ingest.chunk import SampleChunk

# Name of the Apple Watch stream. Confirmed in Phase 0 by moving only the wrist while the
# phone lay still: only this stream showed rotation.
WRIST_MOTION = "wrist motion"

# Value fields of the "wrist motion" stream, as observed in real packets.
# Units follow Apple Core Motion: rotation rate in rad/s, acceleration and gravity in g,
# roll/pitch/yaw in radians.
WRIST_MOTION_FIELDS = (
    "rotationRateX",
    "rotationRateY",
    "rotationRateZ",
    "accelerationX",
    "accelerationY",
    "accelerationZ",
    "gravityX",
    "gravityY",
    "gravityZ",
    "quaternionW",
    "quaternionX",
    "quaternionY",
    "quaternionZ",
    "roll",
    "pitch",
    "yaw",
)

# Unit of every wrist motion field, stored in session metadata so recordings are
# self-describing.
WRIST_MOTION_UNITS = {
    **{f"rotationRate{axis}": "rad/s" for axis in "XYZ"},
    **{f"acceleration{axis}": "g" for axis in "XYZ"},
    **{f"gravity{axis}": "g" for axis in "XYZ"},
    **{f"quaternion{axis}": "1" for axis in "WXYZ"},
    "roll": "rad",
    "pitch": "rad",
    "yaw": "rad",
}


@dataclass(frozen=True)
class SensorSample:
    """One reading from one sensor."""

    name: str
    time_ns: int
    values: dict[str, float]


@dataclass(frozen=True)
class SensorLoggerMessage:
    """One HTTP push message: metadata plus the batch of samples it carries."""

    message_id: int
    session_id: str
    device_id: str
    samples: tuple[SensorSample, ...]


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Sensor Logger message is missing '{key}' in {where}")
    return mapping[key]


def _parse_sample(raw: dict[str, Any], index: int) -> SensorSample:
    where = f"payload[{index}]"
    raw_values = _require(raw, "values", where)
    if not isinstance(raw_values, dict):
        raise ValueError(f"{where}.values must be an object")
    return SensorSample(
        name=str(_require(raw, "name", where)),
        time_ns=int(_require(raw, "time", where)),
        values={key: float(value) for key, value in raw_values.items()},
    )


def parse_message(body: bytes) -> SensorLoggerMessage:
    """Parse one HTTP POST body into a SensorLoggerMessage.

    Raises ValueError with a specific message if the body does not match the observed
    format, so a change in the app's format is noticed instead of silently mis-parsed.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Sensor Logger body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Sensor Logger body must be a JSON object")

    payload = _require(data, "payload", "message")
    if not isinstance(payload, list):
        raise ValueError("Sensor Logger 'payload' must be a list")

    return SensorLoggerMessage(
        message_id=int(_require(data, "messageId", "message")),
        session_id=str(_require(data, "sessionId", "message")),
        device_id=str(_require(data, "deviceId", "message")),
        samples=tuple(_parse_sample(raw, i) for i, raw in enumerate(payload)),
    )


def to_chunk(
    message: SensorLoggerMessage,
    *,
    sensor: str,
    channels: tuple[str, ...],
    packet_seq: int,
    recv_wall_ns: int,
    recv_mono_ns: int,
) -> SampleChunk | None:
    """Extract one sensor's samples from a message as a SampleChunk.

    Returns None if the message has no samples of that sensor (e.g. a packet holding only
    phone sensors). Raises ValueError if a sample lacks one of the requested channels, so a
    renamed field is caught instead of becoming silent zeros.
    """
    selected = [s for s in message.samples if s.name == sensor]
    if not selected:
        return None

    values = np.empty((len(selected), len(channels)), dtype=np.float64)
    for row, sample in enumerate(selected):
        for col, channel in enumerate(channels):
            if channel not in sample.values:
                raise ValueError(f"'{sensor}' sample is missing field '{channel}'")
            values[row, col] = sample.values[channel]

    return SampleChunk(
        packet_seq=packet_seq,
        recv_wall_ns=recv_wall_ns,
        recv_mono_ns=recv_mono_ns,
        time_ns=np.array([s.time_ns for s in selected], dtype=np.int64),
        values=values,
        channels=channels,
    )
