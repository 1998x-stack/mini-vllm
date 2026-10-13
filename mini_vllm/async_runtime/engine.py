from __future__ import annotations

import asyncio
from collections import OrderedDict

from mini_vllm.async_runtime.events import (
    CancelledEvent,
    ErrorEvent,
    FinishedEvent,
    StreamEvent,
    TokenEvent,
)
from mini_vllm.async_runtime.handle import AsyncRequestHandle
from mini_vllm.core.engine import EngineCore
from mini_vllm.core.request import Request


class AsyncEngine:
    """Cooperative asyncio facade over one synchronous EngineCore.

    The facade owns caller handles, pending submissions and bounded delivery
    queues. EngineCore remains the only serving-policy authority.
    """

    def __init__(
        self,
        core: EngineCore,
        *,
        max_queue_size: int = 8,
        version: str = "v0.10",
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        self.core = core
        self.max_queue_size = max_queue_size
        self.version = version
        self._pending: OrderedDict[str, Request] = OrderedDict()
        self._handles: dict[str, AsyncRequestHandle] = {}
        self._seen_request_ids: set[str] = set()
        self._cancel_requested: set[str] = set()
        self._driver_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def driver_running(self) -> bool:
        return self._driver_task is not None and not self._driver_task.done()

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    def get_handle(self, request_id: str) -> AsyncRequestHandle:
        return self._handles[request_id]

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncEngine cannot be driven by multiple event loops")
        return loop

    def _ensure_driver(self) -> None:
        loop = self._bind_loop()
        if self._driver_task is None or self._driver_task.done():
            self._driver_task = loop.create_task(self._drive(), name="mini-vllm-async-driver")

    def submit(self, request: Request) -> AsyncRequestHandle:
        self._bind_loop()
        if request.request_id in self._seen_request_ids:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        handle = AsyncRequestHandle(
            request.request_id,
            max_queue_size=self.max_queue_size,
            cancel_callback=self.cancel,
        )
        self._seen_request_ids.add(request.request_id)
        self._handles[request.request_id] = handle
        self._pending[request.request_id] = request
        self._ensure_driver()
        return handle

    def cancel(self, request_id: str) -> bool:
        handle = self._handles.get(request_id)
        if handle is None or handle.done:
            return False
        self._cancel_requested.add(request_id)
        self._ensure_driver()
        return True

    async def wait_idle(self) -> None:
        while self._driver_task is not None:
            task = self._driver_task
            await asyncio.shield(task)
            if self._driver_task is task and task.done():
                self._driver_task = None

    async def _emit(self, request_id: str, event: StreamEvent) -> None:
        await self._handles[request_id]._emit(event)

    async def _cancel_pending(self) -> None:
        for request_id in tuple(self._cancel_requested):
            request = self._pending.pop(request_id, None)
            if request is None:
                continue
            self._cancel_requested.discard(request_id)
            await self._emit(
                request_id,
                CancelledEvent(request_id, tuple(request.output_token_ids)),
            )

    async def _admit_pending(self) -> None:
        while self._pending:
            request_id, request = self._pending.popitem(last=False)
            if request_id in self._cancel_requested:
                self._cancel_requested.discard(request_id)
                await self._emit(
                    request_id,
                    CancelledEvent(request_id, tuple(request.output_token_ids)),
                )
                continue
            try:
                self.core.add_request(request)
            except Exception as exc:
                await self._emit(
                    request_id,
                    ErrorEvent(request_id, type(exc).__name__, str(exc)),
                )

    async def _cancel_registered(self) -> None:
        for request_id in tuple(self._cancel_requested):
            handle = self._handles.get(request_id)
            if handle is None or handle.done:
                self._cancel_requested.discard(request_id)
                continue
            try:
                request = self.core.get_request(request_id)
            except KeyError:
                continue
            self._cancel_requested.discard(request_id)
            if self.core.cancel_request(request_id):
                await self._emit(
                    request_id,
                    CancelledEvent(request_id, tuple(request.output_token_ids)),
                )

    async def _drain_control(self) -> None:
        # Cancellation wins over admission when both intents are pending at the
        # same cooperative boundary.
        await self._cancel_pending()
        await self._admit_pending()
        await self._cancel_registered()

    async def _publish_step(self, output) -> None:
        for request_id, committed in output.committed_token_ids.items():
            handle = self._handles.get(request_id)
            if handle is None or handle.done:
                continue
            request = self.core.get_request(request_id)
            start_index = len(request.output_token_ids) - len(committed)
            for offset, token_id in enumerate(committed):
                await self._emit(
                    request_id,
                    TokenEvent(request_id, int(token_id), start_index + offset),
                )

        for request_id in output.finished_request_ids:
            handle = self._handles.get(request_id)
            if handle is None or handle.done:
                continue
            request = self.core.get_request(request_id)
            await self._emit(
                request_id,
                FinishedEvent(
                    request_id,
                    tuple(request.output_token_ids),
                    request.finish_reason or "finished",
                ),
            )

    async def _fail_all(self, exc: Exception) -> None:
        self._pending.clear()
        self._cancel_requested.clear()
        for request_id, handle in tuple(self._handles.items()):
            if handle.done:
                continue
            try:
                self.core.abort_request(request_id, reason="error")
            except Exception:
                # Preserve the original execution failure as the caller-facing
                # terminal cause. This is a conservative reference policy.
                pass
            await self._emit(
                request_id,
                ErrorEvent(request_id, type(exc).__name__, str(exc)),
            )

    async def _drive(self) -> None:
        current = asyncio.current_task()
        try:
            while True:
                await self._drain_control()

                if self.core.has_unfinished_requests:
                    try:
                        output = self.core.step()
                    except Exception as exc:
                        await self._fail_all(exc)
                        return
                    await self._publish_step(output)
                    # The current EngineCore step is atomic. Cancellation that
                    # becomes runnable here is applied before the next step.
                    await asyncio.sleep(0)
                    continue

                await self._drain_control()
                if self._pending or self._cancel_requested or self.core.has_unfinished_requests:
                    await asyncio.sleep(0)
                    continue
                return
        finally:
            if self._driver_task is current:
                self._driver_task = None
