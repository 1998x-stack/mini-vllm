from __future__ import annotations

import asyncio
from collections.abc import Callable

from mini_vllm.async_runtime.events import StreamEvent, TerminalEvent, is_terminal_event


class AsyncRequestHandle:
    """Caller-facing bounded stream for one request."""

    def __init__(
        self,
        request_id: str,
        *,
        max_queue_size: int,
        cancel_callback: Callable[[str], bool],
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        self.request_id = request_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._cancel_callback = cancel_callback
        self._terminal_enqueued = False
        self._terminal_event: TerminalEvent | None = None
        self._iterator_exhausted = False
        self._done = asyncio.Event()

    def __aiter__(self) -> AsyncRequestHandle:
        return self

    async def __anext__(self) -> StreamEvent:
        if self._iterator_exhausted:
            raise StopAsyncIteration
        event = await self._queue.get()
        if is_terminal_event(event):
            self._iterator_exhausted = True
        return event

    def cancel(self) -> bool:
        return self._cancel_callback(self.request_id)

    @property
    def done(self) -> bool:
        return self._terminal_enqueued

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        return self._queue.maxsize

    async def wait_terminal(self) -> TerminalEvent:
        await self._done.wait()
        assert self._terminal_event is not None
        return self._terminal_event

    async def _emit(self, event: StreamEvent) -> None:
        if self._terminal_enqueued:
            raise RuntimeError(f"request {self.request_id!r} already has a terminal event")
        await self._queue.put(event)
        if is_terminal_event(event):
            self._terminal_enqueued = True
            self._terminal_event = event
            self._done.set()
