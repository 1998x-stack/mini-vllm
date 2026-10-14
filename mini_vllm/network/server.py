from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import asdict
from typing import TYPE_CHECKING

from mini_vllm.async_runtime import (
    AsyncRequestHandle,
    CancelledEvent,
    ErrorEvent,
    FinishedEvent,
    TokenEvent,
)
from mini_vllm.core.request import Request
from mini_vllm.network.types import GenerateRequest, NetworkLifecycle
from mini_vllm.observability import NetworkSnapshot
from mini_vllm.runtime import NetworkConfig

if TYPE_CHECKING:
    from mini_vllm.versions.v1_2.engine import ArtifactAsyncEngine


_STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    411: "Length Required",
    413: "Content Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class HttpProtocolError(ValueError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _ClientDisconnected(ConnectionError):
    pass


class NetworkServer:
    """Small inspectable HTTP/1.1 + JSON/SSE boundary for v1.3.

    This server intentionally supports one request per TCP connection and
    Content-Length request bodies only. Network admission owns no Scheduler or
    KV policy; after validation it submits an ordinary token-native Request to
    the existing async runtime.
    """

    def __init__(
        self,
        engine: ArtifactAsyncEngine,
        *,
        config: NetworkConfig | None = None,
        owns_engine: bool = False,
    ) -> None:
        self.engine = engine
        self.config = config or NetworkConfig()
        self.owns_engine = owns_engine
        self._state = NetworkLifecycle.CREATED
        self._server: asyncio.AbstractServer | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._inflight: dict[str, AsyncRequestHandle] = {}
        self._disconnect_cancelled_ids: set[str] = set()
        self._request_sequence = 0
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._accepted_requests = 0
        self._rejected_requests = 0
        self._completed_requests = 0
        self._cancelled_requests = 0
        self._failed_requests = 0
        self._disconnect_cancellations = 0

    @property
    def lifecycle_state(self) -> str:
        if self._state is NetworkLifecycle.READY and not self._runtime_ready():
            return "unhealthy"
        return self._state.value

    @property
    def health(self) -> bool:
        return self._state in {NetworkLifecycle.READY, NetworkLifecycle.DRAINING}

    @property
    def ready(self) -> bool:
        return self._state is NetworkLifecycle.READY and self._runtime_ready()

    @property
    def address(self) -> tuple[str, int]:
        if self._bound_host is None or self._bound_port is None:
            raise RuntimeError("network server is not bound")
        return self._bound_host, self._bound_port

    def _runtime_ready(self) -> bool:
        try:
            snapshot = self.engine.snapshot()
        except Exception:
            return False
        artifact = snapshot.core.model_artifact
        return bool(
            snapshot.core.execution.healthy
            and artifact is not None
            and artifact.rank_identity_agreement
        )

    def snapshot(self) -> NetworkSnapshot:
        return NetworkSnapshot(
            lifecycle_state=self.lifecycle_state,
            host=self._bound_host,
            port=self._bound_port,
            health=self.health,
            ready=self.ready,
            active_connections=sum(
                1 for task in self._connection_tasks if not task.done()
            ),
            inflight_requests=len(self._inflight),
            accepted_requests=self._accepted_requests,
            rejected_requests=self._rejected_requests,
            completed_requests=self._completed_requests,
            cancelled_requests=self._cancelled_requests,
            failed_requests=self._failed_requests,
            disconnect_cancellations=self._disconnect_cancellations,
        )

    async def start(self) -> "NetworkServer":
        if self._state is not NetworkLifecycle.CREATED:
            raise RuntimeError(
                f"cannot start network server from state {self._state.value}"
            )
        self._state = NetworkLifecycle.STARTING
        try:
            self._server = await asyncio.start_server(
                self._accept_connection,
                self.config.host,
                self.config.port,
                limit=self.config.max_header_bytes + 1,
            )
            sockets = self._server.sockets or ()
            if not sockets:
                raise RuntimeError("network server started without a bound socket")
            host, port = sockets[0].getsockname()[:2]
            self._bound_host = str(host)
            self._bound_port = int(port)
            self._state = NetworkLifecycle.READY
            return self
        except Exception:
            self._state = NetworkLifecycle.CLOSED
            if self.owns_engine:
                await self.engine.aclose()
            raise

    def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._state is not NetworkLifecycle.READY:
            writer.close()
            return
        task = asyncio.create_task(
            self._handle_connection(reader, writer),
            name="mini-vllm-network-connection",
        )
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    async def close(self) -> None:
        if self._state is NetworkLifecycle.CLOSED:
            return
        self._state = NetworkLifecycle.DRAINING
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for handle in tuple(self._inflight.values()):
            if not handle.done:
                handle.cancel()

        tasks = tuple(task for task in self._connection_tasks if not task.done())
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.shutdown_timeout_s,
                )
            except TimeoutError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        if self.owns_engine:
            await self.engine.aclose()
        self._state = NetworkLifecycle.CLOSED

    async def __aenter__(self) -> "NetworkServer":
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            method, path, headers, body = await self._read_http_request(reader)
            if path in {"/health", "/ready"}:
                if method != "GET":
                    raise HttpProtocolError(405, "method_not_allowed", "endpoint requires GET")
                if body:
                    raise HttpProtocolError(400, "unexpected_body", "GET endpoint does not accept a body")
                if path == "/health":
                    status = 200 if self.health else 503
                    await self._send_json(
                        writer,
                        status,
                        {"status": "ok" if self.health else "unhealthy", "lifecycle": self.lifecycle_state},
                    )
                else:
                    status = 200 if self.ready else 503
                    await self._send_json(
                        writer,
                        status,
                        {"status": "ready" if self.ready else "not_ready", "lifecycle": self.lifecycle_state},
                    )
                return

            if path not in {"/v1/generate", "/v1/generate/stream"}:
                raise HttpProtocolError(404, "not_found", "unknown endpoint")
            if method != "POST":
                raise HttpProtocolError(405, "method_not_allowed", "generation endpoint requires POST")
            content_type = headers.get("content-type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise HttpProtocolError(415, "content_type", "generation body must be application/json")
            request = self._parse_generate_request(body)
            await self._serve_generation(
                request,
                reader=reader,
                writer=writer,
                stream=path.endswith("/stream"),
            )
        except HttpProtocolError as exc:
            self._rejected_requests += 1
            with suppress(_ClientDisconnected):
                await self._send_json(
                    writer,
                    exc.status,
                    {"error": {"type": exc.code, "message": exc.message}},
                )
        except _ClientDisconnected:
            return
        except Exception as exc:
            self._rejected_requests += 1
            with suppress(_ClientDisconnected):
                await self._send_json(
                    writer,
                    500,
                    {"error": {"type": type(exc).__name__, "message": str(exc)}},
                )
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _read_http_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, dict[str, str], bytes]:
        try:
            raw_headers = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError as exc:
            raise HttpProtocolError(431, "headers_too_large", "request headers exceed limit") from exc
        except asyncio.IncompleteReadError as exc:
            raise HttpProtocolError(400, "incomplete_request", "connection ended before headers completed") from exc
        if len(raw_headers) > self.config.max_header_bytes:
            raise HttpProtocolError(431, "headers_too_large", "request headers exceed limit")
        try:
            header_text = raw_headers[:-4].decode("latin-1")
        except UnicodeDecodeError as exc:
            raise HttpProtocolError(400, "invalid_headers", "request headers are not decodable") from exc
        lines = header_text.split("\r\n")
        if not lines or not lines[0]:
            raise HttpProtocolError(400, "invalid_request_line", "missing request line")
        parts = lines[0].split(" ")
        if len(parts) != 3:
            raise HttpProtocolError(400, "invalid_request_line", "malformed request line")
        method, path, version = parts
        if version != "HTTP/1.1":
            raise HttpProtocolError(400, "http_version", "only HTTP/1.1 is supported")

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            if ":" not in line:
                raise HttpProtocolError(400, "invalid_headers", "malformed header line")
            name, value = line.split(":", 1)
            key = name.strip().lower()
            if not key:
                raise HttpProtocolError(400, "invalid_headers", "empty header name")
            if key in headers and key in {"content-length", "transfer-encoding"}:
                raise HttpProtocolError(400, "duplicate_header", f"duplicate {key} header")
            headers[key] = value.strip()

        if "transfer-encoding" in headers:
            raise HttpProtocolError(400, "transfer_encoding", "chunked/transfer-encoded requests are unsupported")

        length_text = headers.get("content-length")
        if method == "POST" and length_text is None:
            raise HttpProtocolError(411, "content_length", "POST requests require Content-Length")
        if length_text is None:
            return method, path, headers, b""
        if not length_text.isdigit():
            raise HttpProtocolError(400, "content_length", "Content-Length must be a non-negative integer")
        length = int(length_text)
        if length > self.config.max_request_bytes:
            raise HttpProtocolError(413, "body_too_large", "request body exceeds configured limit")
        try:
            body = await reader.readexactly(length)
        except asyncio.IncompleteReadError as exc:
            raise HttpProtocolError(400, "incomplete_body", "connection ended before request body completed") from exc
        return method, path, headers, body

    def _next_request_id(self) -> str:
        self._request_sequence += 1
        return f"net-{self._request_sequence:06d}"

    def _parse_generate_request(self, body: bytes) -> GenerateRequest:
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpProtocolError(400, "invalid_json", "request body must contain valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise HttpProtocolError(422, "request_shape", "generation body must be a JSON object")
        allowed = {
            "request_id",
            "prompt",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
        }
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise HttpProtocolError(422, "unknown_field", f"unknown field(s): {', '.join(unexpected)}")
        if "prompt" not in value or "max_new_tokens" not in value:
            raise HttpProtocolError(422, "missing_field", "prompt and max_new_tokens are required")

        request_id = value.get("request_id")
        if request_id is None:
            request_id = self._next_request_id()
        if (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 128
            or any(character.isspace() for character in request_id)
        ):
            raise HttpProtocolError(422, "request_id", "request_id must be 1-128 non-whitespace characters")
        prompt = value["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise HttpProtocolError(422, "prompt", "prompt must be a non-empty string")
        max_new_tokens = value["max_new_tokens"]
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise HttpProtocolError(422, "max_new_tokens", "max_new_tokens must be a positive integer")

        temperature_value = value.get("temperature", 0.0)
        if not isinstance(temperature_value, (int, float)) or isinstance(temperature_value, bool):
            raise HttpProtocolError(422, "temperature", "temperature must be numeric")
        top_k = value.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool)):
            raise HttpProtocolError(422, "top_k", "top_k must be an integer or null")
        top_p_value = value.get("top_p")
        if top_p_value is not None and (
            not isinstance(top_p_value, (int, float)) or isinstance(top_p_value, bool)
        ):
            raise HttpProtocolError(422, "top_p", "top_p must be numeric or null")
        try:
            request = GenerateRequest(
                request_id=request_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature_value),
                top_k=top_k,
                top_p=None if top_p_value is None else float(top_p_value),
            )
            request.sampling_params()
        except ValueError as exc:
            raise HttpProtocolError(422, "sampling_params", str(exc)) from exc
        return request

    def _build_runtime_request(self, request: GenerateRequest) -> Request:
        prompt_token_ids = self.engine.encode(request.prompt)
        if not prompt_token_ids:
            raise HttpProtocolError(422, "prompt_tokens", "prompt produced no token IDs")
        if len(prompt_token_ids) + request.max_new_tokens > self.engine.core.config.max_model_len:
            raise HttpProtocolError(422, "max_model_len", "prompt + max_new_tokens exceeds max_model_len")
        return Request(
            request_id=request.request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=request.max_new_tokens,
            sampling_params=request.sampling_params(),
        )

    async def _serve_generation(
        self,
        request: GenerateRequest,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        stream: bool,
    ) -> None:
        if not self.ready:
            raise HttpProtocolError(503, "not_ready", "runtime is not ready for request admission")
        if len(self._inflight) >= self.config.max_inflight_requests:
            raise HttpProtocolError(429, "ingress_full", "network ingress capacity is full")
        runtime_request = self._build_runtime_request(request)
        try:
            handle = self.engine.submit(runtime_request)
        except ValueError as exc:
            if "duplicate request_id" in str(exc):
                raise HttpProtocolError(409, "duplicate_request_id", str(exc)) from exc
            raise HttpProtocolError(422, "request_rejected", str(exc)) from exc

        self._accepted_requests += 1
        self._inflight[request.request_id] = handle
        disconnect_task: asyncio.Task[None] | None = None
        if self.config.disconnect_cancel:
            disconnect_task = asyncio.create_task(
                self._watch_disconnect(reader, handle),
                name=f"mini-vllm-disconnect-{request.request_id}",
            )
        try:
            if stream:
                await self._stream_generation(writer, handle)
            else:
                await self._complete_generation(writer, handle)
        except _ClientDisconnected:
            self._cancel_for_disconnect(handle)
            await self._consume_to_terminal(handle)
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            self._inflight.pop(request.request_id, None)
            self._disconnect_cancelled_ids.discard(request.request_id)

    async def _watch_disconnect(
        self,
        reader: asyncio.StreamReader,
        handle: AsyncRequestHandle,
    ) -> None:
        try:
            extra = await reader.read(1)
        except (ConnectionError, OSError):
            extra = b""
        if extra == b"":
            self._cancel_for_disconnect(handle)

    def _cancel_for_disconnect(self, handle: AsyncRequestHandle) -> bool:
        if handle.request_id in self._disconnect_cancelled_ids or handle.done:
            return False
        if handle.cancel():
            self._disconnect_cancelled_ids.add(handle.request_id)
            self._disconnect_cancellations += 1
            return True
        return False

    async def _consume_to_terminal(self, handle: AsyncRequestHandle) -> None:
        if handle.done:
            return
        async for event in handle:
            if isinstance(event, (FinishedEvent, CancelledEvent, ErrorEvent)):
                self._record_terminal(event)
                return

    def _record_terminal(self, event: FinishedEvent | CancelledEvent | ErrorEvent) -> None:
        if isinstance(event, FinishedEvent):
            self._completed_requests += 1
        elif isinstance(event, CancelledEvent):
            self._cancelled_requests += 1
        else:
            self._failed_requests += 1

    async def _complete_generation(
        self,
        writer: asyncio.StreamWriter,
        handle: AsyncRequestHandle,
    ) -> None:
        terminal: FinishedEvent | CancelledEvent | ErrorEvent | None = None
        async for event in handle:
            if isinstance(event, (FinishedEvent, CancelledEvent, ErrorEvent)):
                terminal = event
                self._record_terminal(event)
        if terminal is None:
            raise RuntimeError("request stream ended without a terminal event")
        if isinstance(terminal, FinishedEvent):
            await self._send_json(
                writer,
                200,
                {
                    "request_id": terminal.request_id,
                    "output_token_ids": list(terminal.output_token_ids),
                    "text": self.engine.decode(terminal.output_token_ids),
                    "finish_reason": terminal.finish_reason,
                },
            )
        elif isinstance(terminal, CancelledEvent):
            await self._send_json(
                writer,
                503,
                {
                    "error": {
                        "type": "cancelled",
                        "message": "request was cancelled before completion",
                    },
                    "request_id": terminal.request_id,
                    "output_token_ids": list(terminal.output_token_ids),
                },
            )
        else:
            await self._send_json(
                writer,
                500,
                {
                    "error": {"type": terminal.error_type, "message": terminal.message},
                    "request_id": terminal.request_id,
                },
            )

    async def _stream_generation(
        self,
        writer: asyncio.StreamWriter,
        handle: AsyncRequestHandle,
    ) -> None:
        await self._send_sse_headers(writer)
        async for event in handle:
            if isinstance(event, TokenEvent):
                await self._send_sse_event(
                    writer,
                    "token",
                    {
                        "request_id": event.request_id,
                        "token_id": event.token_id,
                        "output_index": event.output_index,
                        "text": self.engine.decode([event.token_id]),
                    },
                )
            elif isinstance(event, FinishedEvent):
                self._record_terminal(event)
                await self._send_sse_event(
                    writer,
                    "done",
                    {
                        "request_id": event.request_id,
                        "output_token_ids": list(event.output_token_ids),
                        "text": self.engine.decode(event.output_token_ids),
                        "finish_reason": event.finish_reason,
                    },
                )
            elif isinstance(event, CancelledEvent):
                self._record_terminal(event)
                await self._send_sse_event(
                    writer,
                    "cancelled",
                    {
                        "request_id": event.request_id,
                        "output_token_ids": list(event.output_token_ids),
                    },
                )
            elif isinstance(event, ErrorEvent):
                self._record_terminal(event)
                await self._send_sse_event(
                    writer,
                    "error",
                    {
                        "request_id": event.request_id,
                        "error_type": event.error_type,
                        "message": event.message,
                    },
                )

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        payload: dict[str, object],
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        reason = _STATUS_TEXT.get(status, "Response")
        headers = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(headers + body)
        await self._drain(writer)

    async def _send_sse_headers(self, writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream; charset=utf-8\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        await self._drain(writer)

    async def _send_sse_event(
        self,
        writer: asyncio.StreamWriter,
        name: str,
        payload: dict[str, object],
    ) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        writer.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
        await self._drain(writer)

    async def _drain(self, writer: asyncio.StreamWriter) -> None:
        try:
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise _ClientDisconnected("client disconnected") from exc
