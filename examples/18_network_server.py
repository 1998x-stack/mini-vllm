from __future__ import annotations

import asyncio
import json
import multiprocessing as mp

from mini_vllm import (
    MultiprocessConfig,
    NetworkConfig,
    RuntimeConfig,
    create_server,
)


START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


async def request(server, method: str, path: str, payload: dict | None = None):
    host, port = server.address
    reader, writer = await asyncio.open_connection(host, port)
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    headers = [
        f"{method} {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Connection: close",
    ]
    if payload is not None:
        headers.extend(
            [
                "Content-Type: application/json",
                f"Content-Length: {len(body)}",
            ]
        )
    writer.write(("\r\n".join(headers) + "\r\n\r\n").encode() + body)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    head, response_body = response.split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    return status, response_body


async def main() -> None:
    config = RuntimeConfig(
        world_size=1,
        multiprocess=MultiprocessConfig(start_method=START_METHOD),
        network=NetworkConfig(host="127.0.0.1", port=0, max_inflight_requests=4),
    )
    server = create_server(runtime_config=config)
    await server.start()
    host, port = server.address
    print(f"server: http://{host}:{port}")

    status, body = await request(server, "GET", "/health")
    print("health:", status, json.loads(body))
    status, body = await request(server, "GET", "/ready")
    print("ready:", status, json.loads(body))

    status, body = await request(
        server,
        "POST",
        "/v1/generate",
        {"request_id": "json-demo", "prompt": "tok1 tok2", "max_new_tokens": 3},
    )
    print("json:", status, json.loads(body))

    status, body = await request(
        server,
        "POST",
        "/v1/generate/stream",
        {"request_id": "sse-demo", "prompt": "tok3 tok4", "max_new_tokens": 3},
    )
    print("sse status:", status)
    print(body.decode("utf-8").strip())

    print("network snapshot:", server.snapshot())
    print("runtime artifact:", server.engine.snapshot().core.model_artifact)
    await server.close()
    print("workers alive after close:", server.engine.core.executor.worker_alive)


if __name__ == "__main__":
    asyncio.run(main())
