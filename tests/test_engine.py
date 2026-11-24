"""Tests for the live pointer loop with a fake clock and a recording mouse."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from airhand.config import DspConfig, IngestConfig, PointerConfig
from airhand.dispatch.windows_input import RecordingMouse, ScreenRect
from airhand.dsp.processor import DspProcessor
from airhand.ingest.chunk import Frames
from airhand.ingest.playout import PlayoutBuffer
from airhand.latency import StageTimer
from airhand.pointer.engine import PointerEngine
from airhand.pointer.processor import PointerProcessor
from airhand.ui.telemetry import TelemetryBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST = IngestConfig.from_yaml(REPO_ROOT / "configs" / "ingest.yaml")
DSP = DspConfig.from_yaml(REPO_ROOT / "configs" / "dsp.yaml")
POINTER = dataclasses.replace(
    PointerConfig.from_yaml(REPO_ROOT / "configs" / "pointer.yaml"), start_engaged=True
)
FS = INGEST.grid_hz
MS = 1_000_000
SCREEN = ScreenRect(0, 0, 4000, 3000)


def turning_frames(n: int) -> Frames:
    """Watch lying flat, turning left and right about the vertical axis."""
    channels = INGEST.resample_channels
    t = np.arange(n)
    values = np.zeros((n, len(channels)))
    values[:, channels.index("rotationRateZ")] = np.radians(80.0) * np.sin(2 * np.pi * t / 300)
    values[:, channels.index("rotationRateY")] = np.radians(30.0) * np.sin(2 * np.pi * t / 170)
    values[:, channels.index("gravityZ")] = -1.0
    values[:, channels.index("quaternionW")] = 1.0
    return Frames(
        time_ns=t.astype(np.int64) * 10 * MS,
        values=values,
        valid=np.ones(n, dtype=bool),
        channels=channels,
    )


def make_engine(hotkey: list[bool] | None = None) -> tuple[PointerEngine, RecordingMouse]:
    mouse = RecordingMouse(SCREEN, x=2000, y=1500)
    presses = iter(hotkey) if hotkey is not None else None
    engine = PointerEngine(
        playout=PlayoutBuffer(
            INGEST.resample_channels,
            10 * MS,
            POINTER.playout_delay_s,
            POINTER.playout_offset_window_s,
        ),
        dsp=DspProcessor(DSP, FS),
        pointer=PointerProcessor(POINTER, FS),
        mouse=mouse,
        timer=StageTimer(),
        telemetry=TelemetryBuffer(capacity=1000),
        hotkey_pressed=(lambda: next(presses, False)) if presses is not None else None,
        tick_hz=100.0,
    )
    return engine, mouse


def test_live_loop_moves_the_cursor_exactly_like_offline_processing() -> None:
    frames = turning_frames(600)
    engine, mouse = make_engine()
    for now_ms in range(0, 8000, 10):
        if now_ms % 1000 == 0 and now_ms < 6000:  # one packet of 100 frames per second
            k = now_ms // 1000
            engine.playout.push(frames.slice(100 * k, 100 * (k + 1)), now_ms * MS)
        engine.tick(now_ms * MS)

    offline = PointerProcessor(POINTER, FS).process(DspProcessor(DSP, FS).process(frames))
    expected_x = 2000 + np.cumsum(offline.dx)
    expected_y = 1500 + np.cumsum(offline.dy)
    assert (mouse.x, mouse.y) == (expected_x[-1], expected_y[-1])
    assert engine.n_frames == 600
    assert len(mouse.path) > 100  # smooth: many small moves, not 6 jumps
    assert engine.playout.n_underruns == 1  # only once the recording runs out

    snap = engine.telemetry.snapshot() if engine.telemetry else None
    assert snap is not None and len(snap.time_s) == 600
    assert snap.status["engaged"] is True
    assert set(snap.status["latency"]) == {"dsp", "pointer", "dispatch"}  # type: ignore[arg-type]


def test_hotkey_press_toggles_the_clutch() -> None:
    engine, mouse = make_engine(hotkey=[True])  # pressed on the first tick
    engine.playout.push(turning_frames(200), 0)
    for now_ms in range(0, 3000, 10):
        engine.tick(now_ms * MS)
    assert engine.pointer.clutch.engaged is False
    assert mouse.path == []
