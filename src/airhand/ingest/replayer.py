"""Replay recorded chunks into the pipeline, in real time or as fast as possible.

The replayer is a drop-in replacement for the live HTTP server: it puts the same kind of
chunks on the same queue, so every downstream stage can be developed and tested without
the watch.

Real-time mode reproduces the original packet timing from the recorded monotonic receive
times. Each chunk is scheduled relative to the start of the replay (not relative to the
previous chunk), so small sleep inaccuracies do not add up over a long session.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from airhand.ingest.chunk import SampleChunk

NS_PER_S = 1_000_000_000

ChunkQueue = asyncio.Queue[SampleChunk | None]


async def replay(
    chunks: list[SampleChunk],
    queue: ChunkQueue,
    *,
    speed: float | None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] | None = None,
) -> None:
    """Put each chunk on the queue, then None to signal the end of the stream.

    Args:
        speed: 1.0 = original timing, 2.0 = twice as fast, None = no waiting at all.
        sleep, clock: Injectable for tests (default: asyncio.sleep and the event loop clock).
    """
    if speed is not None and speed <= 0:
        raise ValueError("speed must be positive, or None for as fast as possible")
    now = clock or asyncio.get_running_loop().time
    start = now()
    first_mono_ns = chunks[0].recv_mono_ns if chunks else 0

    for chunk in chunks:
        if speed is not None:
            due = start + (chunk.recv_mono_ns - first_mono_ns) / NS_PER_S / speed
            delay = due - now()
            if delay > 0:
                await sleep(delay)
        await queue.put(chunk)
    await queue.put(None)
