"""Async HTTP receiver for Sensor Logger pushes.

Each POST is timestamped as soon as its body has been read, parsed, reduced to the
configured sensor's samples, and put on the pipeline queue as one SampleChunk. The server
does no processing beyond that, so a slow downstream stage cannot stall parsing logic, and
the same queue can be fed by the replayer instead.

Responses: 200 "ok" for accepted packets (including packets without the watch sensor,
which are counted and skipped), 400 with the reason for bodies that do not parse.
GET /test answers "ok" so the phone-to-laptop connection can be checked from Safari.
"""

from __future__ import annotations

import asyncio
import time

from aiohttp import web

from airhand.ingest.chunk import SampleChunk
from airhand.ingest.sensor_logger import parse_message, to_chunk


class SensorLoggerReceiver:
    """Turns Sensor Logger HTTP posts into SampleChunks on a queue."""

    def __init__(
        self,
        queue: asyncio.Queue[SampleChunk | None],
        *,
        sensor: str,
        channels: tuple[str, ...],
        data_path: str,
    ) -> None:
        self._queue = queue
        self._sensor = sensor
        self._channels = channels
        self._data_path = data_path
        self._next_seq = 0
        self.n_accepted = 0
        self.n_without_sensor = 0
        self.n_rejected = 0
        # First User-Agent seen (e.g. "SensorLogger/14 ..."); stored in session metadata.
        self.user_agent: str | None = None

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post(self._data_path, self.handle_data)
        app.router.add_get("/test", self.handle_test)
        return app

    async def handle_test(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def handle_data(self, request: web.Request) -> web.Response:
        body = await request.read()
        recv_wall_ns = time.time_ns()
        recv_mono_ns = time.perf_counter_ns()
        packet_seq = self._next_seq
        self._next_seq += 1
        if self.user_agent is None:
            self.user_agent = request.headers.get("User-Agent")

        try:
            message = parse_message(body)
            chunk = to_chunk(
                message,
                sensor=self._sensor,
                channels=self._channels,
                packet_seq=packet_seq,
                recv_wall_ns=recv_wall_ns,
                recv_mono_ns=recv_mono_ns,
            )
        except ValueError as exc:
            self.n_rejected += 1
            return web.Response(status=400, text=str(exc))

        if chunk is None:
            self.n_without_sensor += 1
        else:
            self.n_accepted += 1
            await self._queue.put(chunk)
        return web.Response(text="ok")
