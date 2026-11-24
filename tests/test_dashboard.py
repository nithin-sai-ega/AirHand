"""Smoke test: the dashboard builds and draws a snapshot without a visible window."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from tests.test_engine import make_engine, turning_frames  # noqa: E402

from airhand.ui.dashboard import Dashboard  # noqa: E402
from airhand.ui.telemetry import TelemetryBuffer  # noqa: E402

MS = 1_000_000


def test_dashboard_draws_engine_telemetry() -> None:
    engine, _ = make_engine()
    frames = turning_frames(300)
    for now_ms in range(0, 4000, 10):
        if now_ms % 1000 == 0 and now_ms < 3000:  # one 1 s packet per second
            k = now_ms // 1000
            engine.playout.push(frames.slice(100 * k, 100 * (k + 1)), now_ms * MS)
        engine.tick(now_ms * MS)
    telemetry = engine.telemetry
    assert isinstance(telemetry, TelemetryBuffer)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Dashboard(
        telemetry, refresh_hz=30, window_s=10, knock_threshold_g=0.6, playout_delay_s=0.8
    )
    window.refresh()
    n_frames = len(telemetry.snapshot().time_s)
    assert n_frames == 300
    curve = window.curves["gyro_raw_dps"][0]
    assert len(curve.xData) == n_frames
    assert np.isclose(curve.xData[-1], 0.0)  # newest frame at t = 0
    assert "ENGAGED" in window.status.text()
    window.timer.stop()
    window.close()
    app.processEvents()
