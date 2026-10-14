import asyncio
import json
import multiprocessing as mp

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    MultiprocessConfig,
    NetworkConfig,
    PagedAttentionConfig,
    TinyModelConfig,
    TokenEvent,
    create_engine,
    create_server,
)


FAST_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


def configs():
    return (
        EngineConfig(
            max_num_seqs=2,
            max_model_len=96,
            max_num_batched_tokens=4,
            max_speculative_tokens=1,
        ),
        TinyModelConfig(
            vocab_size=24,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=128,
            seed=53,
        ),
        CacheConfig(block_size=2, num_blocks=64),
    )


def build_engine(version="v1.3", *, max_queue_size=4):
    e, m, c = configs()
    return create_engine(
        version,
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=1,
        max_queue_size=max_queue_size,
        max_speculative_tokens=1,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
        fairness_bypass_limit=2,
        multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
    )


async def raw_http(
    server,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
):
    host, port = server.address
    reader, writer = await asyncio.open_connection(host, port)
    request_headers = {
        "Host": f"{host}:{port}",
        "Connection": "close",
        **(headers or {}),
    }
    if body or method == "POST":
        request_headers.setdefault("Content-Length", str(len(body)))
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(
        f"{name}: {value}\r\n" for name, value in request_headers.items()
    ) + "\r\n"
    writer.write(head.encode("latin-1") + body)
    await writer.drain()
    data = await reader.read()
    writer.close()
    await writer.wait_closed()
    response_head, response_body = data.split(b"\r\n\r\n", 1)
    lines = response_head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    response_headers = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            response_headers[name.lower()] = value.strip()
    return status, response_headers, response_body


async def post_json(server, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    return await raw_http(
        server,
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json"},
    )


def decode_json(body: bytes):
    return json.loads(body.decode("utf-8"))


def parse_sse(body: bytes):
    events = []
    for block in body.decode("utf-8").strip().split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


async def collect_direct(engine, request_id: str, prompt: str, max_new_tokens: int):
    handle = engine.submit_text(
        request_id,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return tokens


def test_health_readiness_snapshot_and_external_engine_ownership() -> None:
    async def scenario():
        engine = build_engine()
        server = engine.create_server(NetworkConfig(port=0))
        await server.start()
        try:
            status, _, body = await raw_http(server, "GET", "/health")
            assert status == 200
            assert decode_json(body)["status"] == "ok"
            status, _, body = await raw_http(server, "GET", "/ready")
            assert status == 200
            assert decode_json(body)["status"] == "ready"

            first = server.snapshot()
            second = server.snapshot()
            assert first == second
            assert first.lifecycle_state == "ready"
            assert first.health and first.ready
            assert first.active_connections == 0
            assert first.inflight_requests == 0
        finally:
            await server.close()

        # A server created around an existing engine does not own that engine.
        assert engine.core.executor.worker_alive == (True,)
        await engine.aclose()
        assert engine.core.executor.worker_alive == (False,)

    asyncio.run(scenario())


def test_json_generation_matches_direct_v12_and_owned_server_reaps_workers() -> None:
    async def scenario():
        prompt = "tok1 tok2 tok3"
        e, m, c = configs()
        reference = build_engine("v1.2")
        try:
            expected = await collect_direct(reference, "direct", prompt, 5)
        finally:
            await reference.aclose()

        server = create_server(
            network_config=NetworkConfig(port=0),
            engine_config=e,
            model_config=m,
            cache_config=c,
            world_size=1,
            max_queue_size=4,
            max_speculative_tokens=1,
            compile_config=CompileConfig(backend="eager"),
            attention_config=PagedAttentionConfig(backend="vectorized"),
            fairness_bypass_limit=2,
            multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
        )
        await server.start()
        engine = server.engine
        try:
            status, headers, body = await post_json(
                server,
                "/v1/generate",
                {
                    "request_id": "http",
                    "prompt": prompt,
                    "max_new_tokens": 5,
                },
            )
            assert status == 200
            assert headers["content-type"].startswith("application/json")
            payload = decode_json(body)
            assert payload["output_token_ids"] == expected
            assert payload["text"] == engine.decode(expected)
            assert payload["finish_reason"] == "length"
            assert server.snapshot().completed_requests == 1
        finally:
            await server.close()
        assert engine.core.executor.worker_alive == (False,)

    asyncio.run(scenario())


def test_sse_stream_contains_committed_tokens_then_one_terminal_event() -> None:
    async def scenario():
        server = create_server(
            network_config=NetworkConfig(port=0),
            engine=build_engine(),
        )
        await server.start()
        engine = server.engine
        try:
            status, headers, body = await post_json(
                server,
                "/v1/generate/stream",
                {
                    "request_id": "stream",
                    "prompt": "tok4 tok5",
                    "max_new_tokens": 4,
                },
            )
            assert status == 200
            assert headers["content-type"].startswith("text/event-stream")
            events = parse_sse(body)
            assert [name for name, _ in events[:-1]] == ["token"] * 4
            assert events[-1][0] == "done"
            token_ids = [payload["token_id"] for name, payload in events if name == "token"]
            assert token_ids == events[-1][1]["output_token_ids"]
            request = engine.core.get_request("stream")
            assert tuple(token_ids) == tuple(request.output_token_ids)
            assert request.finish_reason == "length"
        finally:
            await server.close()
            await engine.aclose()

    asyncio.run(scenario())


def test_invalid_requests_are_rejected_before_runtime_submission() -> None:
    async def scenario():
        engine = build_engine()
        server = engine.create_server(NetworkConfig(port=0, max_request_bytes=64))
        await server.start()
        try:
            status, _, _ = await raw_http(
                server,
                "POST",
                "/v1/generate",
                body=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
            assert status == 400

            status, _, _ = await post_json(
                server,
                "/v1/generate",
                {"prompt": "tok1", "max_new_tokens": 2, "unknown": True},
            )
            assert status == 422

            body = json.dumps({"prompt": "tok1", "max_new_tokens": 2}).encode()
            status, _, _ = await raw_http(
                server,
                "POST",
                "/v1/generate",
                body=body,
                headers={"Content-Type": "text/plain"},
            )
            assert status == 415

            too_large = b"x" * 65
            status, _, _ = await raw_http(
                server,
                "POST",
                "/v1/generate",
                body=too_large,
                headers={"Content-Type": "application/json"},
            )
            assert status == 413

            snapshot = engine.snapshot()
            assert snapshot.pending_request_ids == ()
            assert snapshot.core.requests == ()
            network = server.snapshot()
            assert network.accepted_requests == 0
            assert network.rejected_requests == 4
        finally:
            await server.close()
            await engine.aclose()

    asyncio.run(scenario())


def test_duplicate_request_id_maps_to_conflict_without_second_submission() -> None:
    async def scenario():
        engine = build_engine()
        server = engine.create_server(NetworkConfig(port=0))
        await server.start()
        try:
            payload = {
                "request_id": "dup",
                "prompt": "tok1 tok2",
                "max_new_tokens": 2,
            }
            first, _, _ = await post_json(server, "/v1/generate", payload)
            second, _, body = await post_json(server, "/v1/generate", payload)
            assert first == 200
            assert second == 409
            assert decode_json(body)["error"]["type"] == "duplicate_request_id"
            assert server.snapshot().accepted_requests == 1
        finally:
            await server.close()
            await engine.aclose()

    asyncio.run(scenario())


def test_ingress_capacity_returns_429_without_hidden_runtime_request() -> None:
    async def scenario():
        engine = build_engine(max_queue_size=1)
        server = engine.create_server(
            NetworkConfig(port=0, max_inflight_requests=1)
        )
        await server.start()
        try:
            first_payload = {
                "request_id": "one",
                "prompt": "tok1 tok2 tok3",
                "max_new_tokens": 30,
            }
            second_payload = {
                "request_id": "two",
                "prompt": "tok4 tok5",
                "max_new_tokens": 2,
            }
            results = await asyncio.gather(
                post_json(server, "/v1/generate/stream", first_payload),
                post_json(server, "/v1/generate", second_payload),
            )
            statuses = sorted(result[0] for result in results)
            assert statuses == [200, 429]
            snapshot = server.snapshot()
            assert snapshot.accepted_requests == 1
            assert snapshot.rejected_requests == 1
            assert len(engine.snapshot().core.requests) == 1
        finally:
            await server.close()
            await engine.aclose()

    asyncio.run(scenario())


def test_streaming_disconnect_becomes_cooperative_cancellation() -> None:
    async def scenario():
        engine = build_engine(max_queue_size=1)
        server = engine.create_server(
            NetworkConfig(port=0, disconnect_cancel=True)
        )
        await server.start()
        try:
            host, port = server.address
            reader, writer = await asyncio.open_connection(host, port)
            body = json.dumps(
                {
                    "request_id": "drop",
                    "prompt": "tok1 tok2",
                    "max_new_tokens": 40,
                }
            ).encode()
            request = (
                f"POST /v1/generate/stream HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode() + body
            writer.write(request)
            await writer.drain()
            await reader.readuntil(b"\r\n\r\n")
            first_event = await reader.readuntil(b"\n\n")
            assert b"event: token" in first_event
            writer.close()
            await writer.wait_closed()

            for _ in range(200):
                request_state = engine.core.get_request("drop")
                if request_state.finish_reason == "cancelled":
                    break
                await asyncio.sleep(0.005)
            assert engine.core.get_request("drop").finish_reason == "cancelled"
            assert server.snapshot().disconnect_cancellations == 1
        finally:
            await server.close()
            await engine.aclose()

    asyncio.run(scenario())


def test_header_limit_and_owned_close_with_active_request() -> None:
    async def scenario():
        e, m, c = configs()
        server = create_server(
            network_config=NetworkConfig(
                port=0,
                max_header_bytes=128,
                shutdown_timeout_s=2.0,
            ),
            engine_config=e,
            model_config=m,
            cache_config=c,
            world_size=1,
            max_queue_size=1,
            multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
        )
        await server.start()
        engine = server.engine
        try:
            host, port = server.address
            reader, writer = await asyncio.open_connection(host, port)
            oversized = (
                f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\n"
                f"X-Large: {'x' * 200}\r\n\r\n"
            ).encode()
            writer.write(oversized)
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            assert b" 431 " in response.split(b"\r\n", 1)[0]
        finally:
            await server.close()
        assert server.snapshot().lifecycle_state == "closed"
        assert engine.core.executor.worker_alive == (False,)

    asyncio.run(scenario())
