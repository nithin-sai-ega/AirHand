"""Control the Windows cursor with the watch, live, with the dashboard.

Usage (from the repository root, with the venv active):

    python scripts/pointer_live.py --note "pointer test" [--subject S01 --wrist L]
                                   [--dry-run] [--no-dashboard]

Point Sensor Logger's HTTP push at http://<laptop-ip>:8000/data (see docs/SETUP_NETWORK.md).
The pointer starts OFF. Knock twice on the desk with your knuckles (or press the hotkey, F8 by
default) to toggle it. Every run is also recorded to data/raw/sessions/ so it can be replayed.

--dry-run   run everything but do not move the real cursor
Close the dashboard window or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from airhand.config import DspConfig, IngestConfig, PointerConfig
from airhand.dispatch.windows_input import (
    HotkeyPoller,
    MouseOutput,
    RecordingMouse,
    ScreenRect,
    WindowsMouse,
)
from airhand.dsp.processor import DspProcessor
from airhand.ingest.live import IngestThread, LiveIngest, push_urls
from airhand.ingest.playout import PlayoutBuffer
from airhand.ingest.recorder import SessionRecorder
from airhand.ingest.sensor_logger import WRIST_MOTION_UNITS
from airhand.latency import StageTimer
from airhand.pointer.engine import PointerEngine
from airhand.pointer.processor import PointerProcessor
from airhand.ui.telemetry import TelemetryBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ingest-config", type=Path, default=CONFIGS / "ingest.yaml")
    parser.add_argument("--dsp-config", type=Path, default=CONFIGS / "dsp.yaml")
    parser.add_argument("--pointer-config", type=Path, default=CONFIGS / "pointer.yaml")
    parser.add_argument("--note", default="pointer live", help="What this recording is")
    parser.add_argument("--subject", default=None, help="Anonymous subject id, e.g. S01")
    parser.add_argument("--wrist", choices=["L", "R"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not move the real cursor")
    parser.add_argument("--no-dashboard", action="store_true", help="Console status only")
    return parser.parse_args()


def make_mouse(dry_run: bool) -> MouseOutput:
    if dry_run or sys.platform != "win32":
        return RecordingMouse(ScreenRect(0, 0, 1920, 1080), x=960, y=540)
    return WindowsMouse()


def wait_for_ctrl_c(engine: PointerEngine, live: LiveIngest, interval_s: float) -> None:
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            time.sleep(interval_s)
            clutch = "ON " if engine.pointer.clutch.engaged else "off"
            print(
                f"pointer {clutch} | frames played {engine.n_frames} | "
                f"buffered {engine.playout.buffered_s:.2f} s | "
                f"underruns {engine.playout.n_underruns} | {live.status()}",
                flush=True,
            )


def main() -> int:
    args = parse_args()
    if sys.platform == "win32":
        # Treat Ctrl+Break like Ctrl+C so the recording is also closed cleanly then.
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    ingest_config = IngestConfig.from_yaml(args.ingest_config)
    dsp_config = DspConfig.from_yaml(args.dsp_config)
    pointer_config = PointerConfig.from_yaml(args.pointer_config)
    sample_hz = ingest_config.grid_hz

    timer = StageTimer()
    playout = PlayoutBuffer(
        ingest_config.resample_channels,
        frame_period_ns=round(1e9 / sample_hz),
        delay_s=pointer_config.playout_delay_s,
        offset_window_s=pointer_config.playout_offset_window_s,
    )
    telemetry = TelemetryBuffer(capacity=round(pointer_config.telemetry_window_s * sample_hz))
    hotkey = HotkeyPoller(pointer_config.clutch_hotkey) if sys.platform == "win32" else None
    engine = PointerEngine(
        playout=playout,
        dsp=DspProcessor(dsp_config, sample_hz),
        pointer=PointerProcessor(pointer_config, sample_hz),
        mouse=make_mouse(args.dry_run),
        timer=timer,
        telemetry=telemetry,
        hotkey_pressed=hotkey.pressed if hotkey is not None else None,
        tick_hz=pointer_config.tick_hz,
    )

    session_dir = REPO_ROOT / ingest_config.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    info = {
        "note": args.note,
        "subject_id": args.subject,
        "wrist": args.wrist,
        "source_app": ingest_config.source_app,
        "sensor": ingest_config.sensor,
        "origin": "pointer_live",
    }
    with SessionRecorder(
        session_dir,
        channels=ingest_config.record_channels,
        units=WRIST_MOTION_UNITS,
        info=info,
        flush_interval_s=ingest_config.flush_interval_s,
    ) as recorder:
        live = LiveIngest(
            ingest_config,
            recorder,
            on_frames=lambda frames: playout.push(frames, time.monotonic_ns()),
            timer=timer,
            print_status=False,
        )
        ingest = IngestThread(live)
        ingest.start()
        engine.start()
        print(f"Recording to {session_dir}")
        for url in push_urls(ingest_config):
            print(f"  Sensor Logger push URL: {url}")
        print(
            f"Pointer is OFF. Double knock or press {pointer_config.clutch_hotkey} to toggle."
            + (" (dry run: the real cursor will not move)" if args.dry_run else "")
        )
        try:
            if args.no_dashboard:
                print("Press Ctrl+C to stop.", flush=True)
                wait_for_ctrl_c(engine, live, ingest_config.stats_interval_s)
            else:
                run_with_dashboard(telemetry, pointer_config)
        finally:
            engine.stop()
            ingest.stop()
            print(live.status())
    print(f"Saved {recorder.n_samples} samples to {session_dir}")
    return 0


def run_with_dashboard(telemetry: TelemetryBuffer, config: PointerConfig) -> None:
    # Imported here so --no-dashboard works without loading Qt.
    from PySide6 import QtWidgets

    from airhand.ui.dashboard import Dashboard, run_dashboard

    assert threading.current_thread() is threading.main_thread(), "Qt needs the main thread"
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = Dashboard(
        telemetry,
        refresh_hz=config.dashboard_refresh_hz,
        window_s=config.telemetry_window_s,
        knock_threshold_g=config.knock_threshold_g,
        playout_delay_s=config.playout_delay_s,
    )
    run_dashboard(window)


if __name__ == "__main__":
    raise SystemExit(main())
