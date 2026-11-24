"""The live pointer loop.

Every tick (100 Hz by default):
    1. poll the clutch hotkey
    2. take the frames whose play time has come from the playout buffer
    3. DSP -> pointer -> one combined cursor move for this tick
    4. update the dashboard telemetry and the per-stage timings

`tick(now_ns)` does one step and is what the tests call with a fake clock; `run()` repeats it
on the real monotonic clock until `stop()` is called. It runs in its own thread so the
dashboard window stays responsive.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from airhand.dispatch.windows_input import MouseOutput
from airhand.dsp.processor import DspProcessor
from airhand.ingest.playout import PlayoutBuffer
from airhand.latency import StageTimer
from airhand.pointer.processor import PointerProcessor
from airhand.ui.telemetry import TelemetryBuffer

NS_PER_S = 1_000_000_000


class PointerEngine:
    def __init__(
        self,
        *,
        playout: PlayoutBuffer,
        dsp: DspProcessor,
        pointer: PointerProcessor,
        mouse: MouseOutput,
        timer: StageTimer,
        telemetry: TelemetryBuffer | None = None,
        hotkey_pressed: Callable[[], bool] | None = None,
        tick_hz: float = 100.0,
    ) -> None:
        self.playout = playout
        self.dsp = dsp
        self.pointer = pointer
        self.mouse = mouse
        self.timer = timer
        self.telemetry = telemetry
        self.hotkey_pressed = hotkey_pressed
        self.tick_ns = round(NS_PER_S / tick_hz)
        self.n_frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self, now_ns: int) -> None:
        if self.hotkey_pressed is not None and self.hotkey_pressed():
            self.pointer.toggle_clutch()

        frames = self.playout.pop_due(now_ns)
        if len(frames) > 0:
            with self.timer.measure("dsp"):
                dsp_out = self.dsp.process(frames)
            with self.timer.measure("pointer"):
                pointer_out = self.pointer.process(dsp_out)
            with self.timer.measure("dispatch"):
                self.mouse.move_by(int(pointer_out.dx.sum()), int(pointer_out.dy.sum()))
            self.n_frames += len(frames)
            if self.telemetry is not None:
                self.telemetry.append(dsp_out, pointer_out)

        if self.telemetry is not None:
            self.telemetry.set_status(
                engaged=self.pointer.clutch.engaged,
                toggles=self.pointer.clutch.n_toggles,
                buffered_s=self.playout.buffered_s,
                starved=self.playout.starved,
                underruns=self.playout.n_underruns,
                frames=self.n_frames,
                latency=self.timer.summary(),
            )

    def run(self) -> None:
        """Tick at a fixed rate until stop() is called."""
        next_tick = time.monotonic_ns()
        while not self._stop.is_set():
            self.tick(time.monotonic_ns())
            next_tick += self.tick_ns
            delay_ns = next_tick - time.monotonic_ns()
            if delay_ns > 0:
                time.sleep(delay_ns / NS_PER_S)
            else:
                next_tick = time.monotonic_ns()  # fell behind: do not try to catch up

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="pointer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
