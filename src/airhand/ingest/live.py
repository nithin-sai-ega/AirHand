"""Live ingest: HTTP receiver + pipeline, as an asyncio service and as a background thread.

`LiveIngest` has an explicit start / stop instead of a `finally` inside a long-running
coroutine: when Ctrl+C interrupts the event loop, `stop()` is then run on the same loop, so
the queue is always drained and the recording's metadata completed.

`IngestThread` runs a LiveIngest on its own event loop in a background thread. The pointer
needs this because its main thread is busy with the dashboard window (Qt) and its own loop.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable

from aiohttp import web

from airhand.config import IngestConfig
from airhand.ingest.chunk import Frames, SampleChunk
from airhand.ingest.http_server import SensorLoggerReceiver
from airhand.ingest.pipeline import IngestPipeline, build_pipeline
from airhand.ingest.recorder import SessionRecorder
from airhand.latency import StageTimer


def lan_ipv4_addresses() -> list[str]:
    try:
        _, _, addresses = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return []
    return sorted(a for a in addresses if not a.startswith("127."))


def push_urls(config: IngestConfig) -> list[str]:
    return [
        f"http://{ip}:{config.http_port}{config.data_path}"
        for ip in lan_ipv4_addresses() or ["<laptop-ip>"]
    ]


def status_line(pipeline: IngestPipeline, receiver: SensorLoggerReceiver) -> str:
    s = pipeline.stats
    age = f"{s.last_sample_age_ms:.0f} ms" if s.last_sample_age_ms is not None else "-"
    return (
        f"packets {s.chunks} | samples {s.samples} | frames {s.frames} "
        f"(invalid {s.invalid_frames}) | dropped {pipeline.n_dropped_samples} | "
        f"rejected {receiver.n_rejected} | no-watch {receiver.n_without_sensor} | "
        f"newest sample age {age} | buffer {len(pipeline.ring)}/{pipeline.ring.capacity}"
    )


class LiveIngest:
    def __init__(
        self,
        config: IngestConfig,
        recorder: SessionRecorder,
        *,
        on_frames: Callable[[Frames], None] | None = None,
        timer: StageTimer | None = None,
        print_status: bool = True,
    ) -> None:
        self.config = config
        self.recorder = recorder
        self.print_status = print_status
        self.queue: asyncio.Queue[SampleChunk | None] = asyncio.Queue(
            maxsize=config.queue_max_chunks
        )
        self.receiver = SensorLoggerReceiver(
            self.queue,
            sensor=config.sensor,
            channels=config.record_channels,
            data_path=config.data_path,
        )
        self.pipeline = build_pipeline(config, recorder, on_frames, timer)
        self._runner = web.AppRunner(self.receiver.make_app())
        self._pipeline_task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._runner.setup()
        await web.TCPSite(self._runner, self.config.bind_host, self.config.http_port).start()
        self._pipeline_task = asyncio.create_task(self.pipeline.run(self.queue))
        if self.print_status:
            self._report_task = asyncio.create_task(self._report())

    async def _report(self) -> None:
        while True:
            await asyncio.sleep(self.config.stats_interval_s)
            print(self.status(), flush=True)

    def status(self) -> str:
        return status_line(self.pipeline, self.receiver)

    async def wait_forever(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        """Stop accepting packets, process everything already queued, complete metadata."""
        if self._report_task is not None:
            self._report_task.cancel()
        await self._runner.cleanup()
        await self.queue.put(None)
        if self._pipeline_task is not None:
            await self._pipeline_task
        self.recorder.update_info({"user_agent": self.receiver.user_agent})


class IngestThread:
    """Runs a LiveIngest on a private event loop in a daemon thread."""

    def __init__(self, live: LiveIngest) -> None:
        self.live = live
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="ingest", daemon=True)

    def start(self) -> None:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.live.start(), self.loop).result()

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.live.stop(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join()
        self.loop.close()
