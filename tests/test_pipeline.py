"""End-to-end ingest tests: HTTP -> queue -> pipeline -> recorder, and replay determinism."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from airhand.config import IngestConfig
from airhand.ingest.chunk import Frames, SampleChunk
from airhand.ingest.http_server import SensorLoggerReceiver
from airhand.ingest.live import IngestThread, LiveIngest
from airhand.ingest.pipeline import build_pipeline
from airhand.ingest.recorder import SessionRecorder, read_session
from airhand.ingest.replayer import replay
from airhand.ingest.resampler import resample_offline
from airhand.ingest.sensor_logger import WRIST_MOTION, WRIST_MOTION_UNITS
from airhand.latency import StageTimer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = IngestConfig.from_yaml(REPO_ROOT / "configs" / "ingest.yaml")
MS = 1_000_000
T0 = 1_795_463_355_248_168_700
PERIOD_NS = 10_010_000  # the watch's real 10.01 ms spacing


def watch_like_chunks(n_packets: int, seed: int = 0) -> list[SampleChunk]:
    """Packets shaped like Phase 0 data: 102 or 153 samples at 99.9 Hz, one gap inside."""
    rng = np.random.default_rng(seed)
    channels = CONFIG.record_channels
    chunks = []
    k = 0
    for seq in range(n_packets):
        n = int(rng.choice([102, 153]))
        idx = np.arange(k, k + n)
        k += n
        times = T0 + idx * PERIOD_NS
        if seq == n_packets // 2:
            times = times + 200 * MS  # a 200 ms hole before this packet
        values = rng.normal(size=(n, len(channels)))
        quat = values[:, 9:13]
        values[:, 9:13] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        chunks.append(
            SampleChunk(
                packet_seq=seq,
                recv_wall_ns=int(times[-1]) + 800 * MS,
                recv_mono_ns=seq * 1_100 * MS,
                time_ns=times.astype(np.int64),
                values=values,
                channels=channels,
            )
        )
    return chunks


def replay_through_pipeline(chunks: list[SampleChunk]) -> tuple[str, Frames, int]:
    """Replay as fast as possible; return digest, all frames, and invalid frame count."""
    pipeline = build_pipeline(CONFIG)
    collected: list[Frames] = []
    original_process = pipeline.process

    def process_and_collect(chunk: SampleChunk) -> Frames:
        frames = original_process(chunk)
        collected.append(frames)
        return frames

    pipeline.process = process_and_collect  # type: ignore[method-assign]

    async def main() -> None:
        queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue()
        await asyncio.gather(replay(chunks, queue, speed=None), pipeline.run(queue))

    asyncio.run(main())
    frames = Frames.concatenate(collected, CONFIG.resample_channels)
    return pipeline.digest(), frames, pipeline.stats.invalid_frames


def test_replay_is_deterministic_and_matches_offline_resampling(tmp_path: Path) -> None:
    chunks = watch_like_chunks(40)
    with SessionRecorder(
        tmp_path / "session",
        channels=CONFIG.record_channels,
        units=WRIST_MOTION_UNITS,
        info={"note": "determinism test"},
        flush_interval_s=CONFIG.flush_interval_s,
    ) as recorder:
        for chunk in chunks:
            recorder.append(chunk)
    recorded = read_session(tmp_path / "session").chunks

    digest_1, frames_1, invalid_1 = replay_through_pipeline(recorded)
    digest_2, frames_2, _ = replay_through_pipeline(recorded)
    assert digest_1 == digest_2, "same input must give identical output"

    offline = resample_offline(recorded, build_pipeline(CONFIG).resampler)
    np.testing.assert_array_equal(frames_1.time_ns, offline.time_ns)
    np.testing.assert_array_equal(frames_1.values, offline.values)
    np.testing.assert_array_equal(frames_2.valid, offline.valid)
    # Every grid point strictly inside the hole (and only those) must be invalid.
    hole_start = int(recorded[len(recorded) // 2 - 1].time_ns[-1])
    hole_end = int(recorded[len(recorded) // 2].time_ns[0])
    inside = (frames_1.time_ns > hole_start) & (frames_1.time_ns < hole_end)
    assert inside.sum() == 21  # 210.01 ms hole on a 10 ms grid
    np.testing.assert_array_equal(~frames_1.valid, inside)
    assert invalid_1 == inside.sum()


def test_ring_buffer_holds_the_newest_frames() -> None:
    chunks = watch_like_chunks(30)
    pipeline = build_pipeline(CONFIG)
    all_frames = Frames.concatenate([pipeline.process(c) for c in chunks], CONFIG.resample_channels)
    tail = pipeline.ring.latest(CONFIG.ring_capacity_frames)
    np.testing.assert_array_equal(tail.time_ns, all_frames.time_ns[-len(tail) :])
    assert len(tail) == min(len(all_frames), CONFIG.ring_capacity_frames)


def sensor_logger_body(message_id: int, samples: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "messageId": message_id,
            "sessionId": "00000000-0000-4000-8000-00000000000a",
            "deviceId": "00000000-0000-4000-8000-00000000000b",
            "payload": samples,
        }
    ).encode()


def wrist(time_ns: int, value: float) -> dict[str, Any]:
    values = {name: value for name in CONFIG.record_channels}
    values.update(quaternionW=1.0, quaternionX=0.0, quaternionY=0.0, quaternionZ=0.0)
    return {"name": WRIST_MOTION, "time": time_ns, "values": values}


def test_http_posts_flow_through_pipeline_into_recording(tmp_path: Path) -> None:
    async def main() -> tuple[list[int], SensorLoggerReceiver]:
        queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue()
        receiver = SensorLoggerReceiver(
            queue, sensor=CONFIG.sensor, channels=CONFIG.record_channels, data_path=CONFIG.data_path
        )
        bodies = [
            sensor_logger_body(1, [wrist(T0 + i * PERIOD_NS, float(i)) for i in range(5)]),
            sensor_logger_body(2, [{"name": "gyroscope", "time": T0, "values": {"x": 0.0}}]),
            b"{not json",
            sensor_logger_body(3, [wrist(T0 + i * PERIOD_NS, float(i)) for i in range(5, 9)]),
        ]
        statuses = []
        async with TestClient(TestServer(receiver.make_app())) as client:
            for body in bodies:
                response = await client.post(
                    CONFIG.data_path, data=body, headers={"User-Agent": "SensorLogger/14"}
                )
                statuses.append(response.status)
            test_response = await client.get("/test")
            assert test_response.status == 200 and await test_response.text() == "ok"
        await queue.put(None)

        with SessionRecorder(
            tmp_path / "live",
            channels=CONFIG.record_channels,
            units=WRIST_MOTION_UNITS,
            info={"user_agent": receiver.user_agent},
            flush_interval_s=CONFIG.flush_interval_s,
        ) as recorder:
            await build_pipeline(CONFIG, recorder).run(queue)
        return statuses, receiver

    statuses, receiver = asyncio.run(main())

    assert statuses == [200, 200, 400, 200]
    assert (receiver.n_accepted, receiver.n_without_sensor, receiver.n_rejected) == (2, 1, 1)
    session = read_session(tmp_path / "live")
    assert session.meta["user_agent"] == "SensorLogger/14"
    assert [c.packet_seq for c in session.chunks] == [0, 3]
    times = np.concatenate([c.time_ns for c in session.chunks])
    np.testing.assert_array_equal(times, T0 + np.arange(9) * PERIOD_NS)
    rotation_x = np.concatenate([c.values[:, 0] for c in session.chunks])
    np.testing.assert_array_equal(rotation_x, np.arange(9.0))


def test_pipeline_hands_new_frames_to_the_callback_and_records_timings() -> None:
    received: list[Frames] = []
    timer = StageTimer()
    pipeline = build_pipeline(CONFIG, on_frames=received.append, timer=timer)
    chunks = watch_like_chunks(4)
    produced = [pipeline.process(chunk) for chunk in chunks]
    assert len(received) == sum(1 for f in produced if len(f) > 0)
    np.testing.assert_array_equal(
        Frames.concatenate(received, CONFIG.resample_channels).time_ns,
        Frames.concatenate(produced, CONFIG.resample_channels).time_ns,
    )
    summary = timer.summary()
    assert summary["ingest"]["n"] == 4
    assert summary["sample_age"]["p50"] == pytest.approx(800.0)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_ingest_thread_receives_http_posts_in_the_background(tmp_path: Path) -> None:
    config = dataclasses.replace(CONFIG, bind_host="127.0.0.1", http_port=free_port())
    received: list[Frames] = []
    with SessionRecorder(
        tmp_path / "bg",
        channels=config.record_channels,
        units=WRIST_MOTION_UNITS,
        info={},
        flush_interval_s=config.flush_interval_s,
    ) as recorder:
        thread = IngestThread(
            LiveIngest(config, recorder, on_frames=received.append, print_status=False)
        )
        thread.start()
        try:
            body = sensor_logger_body(1, [wrist(T0 + i * PERIOD_NS, float(i)) for i in range(50)])
            url = f"http://127.0.0.1:{config.http_port}{config.data_path}"
            with urllib.request.urlopen(urllib.request.Request(url, data=body)) as response:
                assert response.status == 200
        finally:
            thread.stop()
    assert sum(len(f) for f in received) > 0
    assert read_session(tmp_path / "bg").meta["n_samples"] == 50
