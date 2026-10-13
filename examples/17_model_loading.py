from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
from pathlib import Path

from mini_vllm import (
    MultiprocessConfig,
    TinyModelConfig,
    TokenEvent,
    create_engine,
    export_tiny_model_artifact,
)


async def main() -> None:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with tempfile.TemporaryDirectory(prefix="mini-vllm-example-") as directory:
        root = Path(directory)
        artifact = export_tiny_model_artifact(
            root,
            TinyModelConfig(
                vocab_size=24,
                d_model=16,
                n_heads=4,
                n_kv_heads=2,
                n_layers=1,
                d_ff=32,
                max_position_embeddings=64,
                seed=41,
            ),
        )
        print("artifact:", artifact.identity.digest[:16])
        print("weights:", len(artifact.weight_names))

        engine = create_engine(
            "v1.2",
            model_artifact=str(root),
            world_size=2,
            multiprocess_config=MultiprocessConfig(start_method=start_method),
        )
        try:
            prompt = "tok1 tok2 tok3 tok4 tok5"
            print("text:", prompt)
            print("token ids:", engine.encode(prompt))
            handle = engine.submit_text("demo", prompt, max_new_tokens=3)
            output = []
            async for event in handle:
                if isinstance(event, TokenEvent):
                    output.append(event.token_id)
            await engine.wait_idle()
            print("output ids:", output)
            print("decoded:", engine.decode(output))

            snapshot = engine.snapshot()
            metadata = snapshot.core.model_artifact
            assert metadata is not None
            print("snapshot artifact:", metadata.identity[:16])
            print("rank identity agreement:", metadata.rank_identity_agreement)
            print(
                "worker pids:",
                [rank.process_id for rank in snapshot.core.execution.ranks],
            )
        finally:
            await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
